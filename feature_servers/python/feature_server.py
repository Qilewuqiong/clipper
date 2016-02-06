#!/usr/bin/env python

from __future__ import print_function, absolute_import

import findspark
findspark.init()

import pyspark
import sys
import time
import os
import numpy
from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionModel, LogisticRegressionWithSGD

import argparse
import socket
import random
import capnp
import os
from sklearn.externals import joblib
import numpy as np
# from collections import OrderedDict
from sklearn import linear_model as lm
import sklearn.svm as svm
capnp.remove_import_hook()
feature_capnp = capnp.load(os.path.abspath('../../clipper_server/schema/feature.capnp'))
from sample_feature import TestFeature





# def evaluate_impl(expression, params=None):
#     '''Implementation of CalculatorImpl::evaluate(), also shared by
#     FunctionImpl::call().  In the latter case, `params` are the parameter
#     values passed to the function; in the former case, `params` is just an
#     empty list.'''
#
#     which = expression.which()
#
#     if which == 'literal':
#         return capnp.Promise(expression.literal)
#     elif which == 'previousResult':
#         return read_value(expression.previousResult)
#     elif which == 'parameter':
#         assert expression.parameter < len(params)
#         return capnp.Promise(params[expression.parameter])
#     elif which == 'call':
#         call = expression.call
#         func = call.function
#
#         # Evaluate each parameter.
#         paramPromises = [evaluate_impl(param, params) for param in call.params]
#
#         joinedParams = capnp.join_promises(paramPromises)
#         # When the parameters are complete, call the function.
#         ret = (joinedParams
#                .then(lambda vals: func.call(vals))
#                .then(lambda result: result.value))
#
#         return ret
#     else:
#         raise ValueError("Unknown expression type: " + which)
#
#
# class ValueImpl(calculator_capnp.Calculator.Value.Server):
#
#     "Simple implementation of the Calculator.Value Cap'n Proto interface."
#
#     def __init__(self, value):
#         self.value = value
#
#     def read(self, **kwargs):
#         return self.value
#
#
# class FunctionImpl(calculator_capnp.Calculator.Function.Server):
#
#     '''Implementation of the Calculator.Function Cap'n Proto interface, where the
#     function is defined by a Calculator.Expression.'''
#
#     def __init__(self, paramCount, body):
#         self.paramCount = paramCount
#         self.body = body.as_builder()
#
#     def call(self, params, _context, **kwargs):
#         '''Note that we're returning a Promise object here, and bypassing the
#         helper functionality that normally sets the results struct from the
#         returned object. Instead, we set _context.results directly inside of
#         another promise'''
#
#         assert len(params) == self.paramCount
#         # using setattr because '=' is not allowed inside of lambdas
#         return evaluate_impl(self.body, params).then(lambda value: setattr(_context.results, 'value', value))
#
#
# class OperatorImpl(calculator_capnp.Calculator.Function.Server):
#
#     '''Implementation of the Calculator.Function Cap'n Proto interface, wrapping
#     basic binary arithmetic operators.'''
#
#     def __init__(self, op):
#         self.op = op
#
#     def call(self, params, **kwargs):
#         assert len(params) == 2
#
#         op = self.op
#
#         if op == 'add':
#             return params[0] + params[1]
#         elif op == 'subtract':
#             return params[0] - params[1]
#         elif op == 'multiply':
#             return params[0] * params[1]
#         elif op == 'divide':
#             return params[0] / params[1]
#         else:
#             raise ValueError('Unknown operator')
#
#
# class CalculatorImpl(calculator_capnp.Calculator.Server):
#
#     "Implementation of the Calculator Cap'n Proto interface."
#
#     def evaluate(self, expression, _context, **kwargs):
#         return evaluate_impl(expression).then(lambda value: setattr(_context.results, 'value', ValueImpl(value)))
#
#     def defFunction(self, paramCount, body, _context, **kwargs):
#         return FunctionImpl(paramCount, body)
#
#     def getOperator(self, op, **kwargs):
#         return OperatorImpl(op)

def load_scikit_model(pickle_path):
    # pickle_loc = "%s/features/%s/%s.pkl" % (self.namespace, lf, lf)
    feature = joblib.load(pickle_path)
    name = os.path.basename(pickle_path).strip(".pkl")
    return (name, feature)



class ScikitFeatureImpl(feature_capnp.Feature.Server):
    

    def __init__(self, path):
        self.name, self.model = load_scikit_model(path)
        print("started sklearn")


    # def load_feature_functions(self):
    #     feature_objects = []
    #     feature_names = [line.strip() for line in open(self.model_path, 'r')]
    #     for lf in feature_names:
    #         pickle_loc = "%s/features/%s/%s.pkl" % (self.namespace, lf, lf)
    #         feature = joblib.load(pickle_loc) 
    #         feature_objects.append(feature)
    #     return (feature_objects, feature_names)

    def computeFeature(self, inp, _context, **kwargs):
        # print(_context.params)
        # s = name
        print(type(self.model))
        pred = self.model.predict(np.array(inp).reshape(1, -1))[0]
        print("Model predicted: %f" % pred)
        return float(pred)

class PySparkFeatureImpl(feature_capnp.Feature.Server):
    # TODO: find out how to stop spark context
    def __init__(self, path):

        conf = SparkConf() \
            .setAppName("crankshaw-pyspark") \
            .set("spark.executor.memory", "2g") \
            .set("spark.kryoserializer.buffer.mb", "128") \
            .set("master", "local")
        sc = SparkContext(conf=conf, batchSize=10)
        self.model = LogisticRegressionModel.load(sc, path)
        # path = '/Users/crankshaw/model-serving/tugboat/feature_servers/python/spark_model'
        # self.name, self.model = load_pyspark_model(path)

        print("started spark")

    def computeFeature(self, inp, _context, **kwargs):
        # print(_context.params)
        # s = name
        # print(type(self.model))
        x = [float(v)/255.0 for v in inp]
        pred = self.model.predict(x)
        print("GREPTHIS Model predicted: %f" % pred)
        return float(pred)

    # def load_pyspark_model(self, sc, path):
    #     # pickle_loc = "%s/features/%s/%s.pkl" % (self.namespace, lf, lf)
    #
    #     sameModel = LogisticRegressionModel.load(sc, path)
    #     feature = joblib.load(pickle_path)
    #     name = os.path.basename(pickle_path).strip(".pkl")
    #     return (name, feature)

def parse_args():
    parser = argparse.ArgumentParser(usage='''Runs the server bound to the\
given address/port ADDRESS may be '*' to bind to all local addresses.\
:PORT may be omitted to choose a port automatically. ''')

    parser.add_argument("address", type=str, help="ADDRESS[:PORT]")
    parser.add_argument("framework", type=str, help="spark|sklearn")
    parser.add_argument("modelpath", help="full path to pickled model file")


    return parser.parse_args()


def main():
    args = parse_args()
    address = args.address
    model_path = args.modelpath
    # print(model_path)
    if args.framework == "spark":
        server = capnp.TwoPartyServer(address, bootstrap=PySparkFeatureImpl(model_path))
    elif args.framework == "sklearn":
        server = capnp.TwoPartyServer(address, bootstrap=ScikitFeatureImpl(model_path))
    else:
        print("%s is unsupported framework" % args.system)
        return
    server.run_forever()

if __name__ == '__main__':
    main()


