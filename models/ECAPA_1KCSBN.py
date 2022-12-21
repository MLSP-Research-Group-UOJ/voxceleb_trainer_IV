#! /usr/bin/python
# -*- encoding: utf-8 -*-

from models.ECAPA_TDNN import *


def MainModel(**kwargs):
    model = Res2Net(Bottle2neck, C=1024, model_scale=8, context=True, summed=True, out_bn=True, **kwargs)
    return model
