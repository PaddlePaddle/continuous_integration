#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
  * @file test_static_metrics.py
  * @author zhengtianyu
  * @date 2019/11/6 2:49 下午
  * @brief 
  *
  **************************************************************************/
"""

from mnist import MNIST
import paddle.fluid as fluid
import paddle
import numpy as np
import tools


def test_accuracy():
    """
    test accuacy
    :return:
    """
    import paddle.fluid as fluid
    # 假设有batch_size = 128
    batch_size = 128
    accuracy_manager = fluid.metrics.Accuracy()
    # 假设第一个batch的准确率为0.9
    batch1_acc = 0.9
    accuracy_manager.update(value=batch1_acc, weight=batch_size)
    print("expect accuracy: %.2f, get accuracy: %.2f" %
          (batch1_acc, accuracy_manager.eval()))
    acc = batch1_acc
    tools.compare(acc, accuracy_manager.eval())
    # 假设第二个batch的准确率为0.8
    batch2_acc = 0.8
    accuracy_manager.update(value=batch2_acc, weight=batch_size)
    # batch1和batch2的联合准确率为(batch1_acc * batch_size + batch2_acc * batch_size) / batch_size / 2
    print("expect accuracy: %.2f, get accuracy: %.2f" %
          ((batch1_acc * batch_size + batch2_acc * batch_size) / batch_size / 2,
           accuracy_manager.eval()))
    acc = (batch1_acc * batch_size + batch2_acc * batch_size) / batch_size / 2
    tools.compare(acc, accuracy_manager.eval())
    # 假设第三个batch的准确率为0.8
    batch3_acc = 0.8
    accuracy_manager.update(value=batch3_acc, weight=batch_size)
    print("expect accuracy: %.2f, get accuracy: %.2f" % (
        (batch1_acc * batch_size + batch2_acc * batch_size + batch3_acc *
         batch_size) / batch_size / 3, accuracy_manager.eval()))
    acc = (batch1_acc * batch_size + batch2_acc * batch_size + batch3_acc *
           batch_size) / batch_size / 3
    tools.compare(acc, accuracy_manager.eval())


def test_accuracy1():
    """
    test accuacy with reset
    :return:
    """
    import paddle.fluid as fluid
    # 假设有batch_size = 128
    batch_size = 128
    accuracy_manager = fluid.metrics.Accuracy()
    # 假设第一个batch的准确率为0.9
    batch1_acc = 0.9
    accuracy_manager.update(value=batch1_acc, weight=batch_size)
    print("expect accuracy: %.2f, get accuracy: %.2f" %
          (batch1_acc, accuracy_manager.eval()))
    acc = batch1_acc
    tools.compare(acc, accuracy_manager.eval())
    # 假设第二个batch的准确率为0.8
    batch2_acc = 0.8
    accuracy_manager.update(value=batch2_acc, weight=batch_size)
    # batch1和batch2的联合准确率为(batch1_acc * batch_size + batch2_acc * batch_size) / batch_size / 2
    print("expect accuracy: %.2f, get accuracy: %.2f" %
          ((batch1_acc * batch_size + batch2_acc * batch_size) / batch_size / 2,
           accuracy_manager.eval()))
    acc = (batch1_acc * batch_size + batch2_acc * batch_size) / batch_size / 2
    tools.compare(acc, accuracy_manager.eval())
    # 重置accuracy_manager
    accuracy_manager.reset()
    # 假设第三个batch的准确率为0.8
    batch3_acc = 0.8
    accuracy_manager.update(value=batch3_acc, weight=batch_size)
    print("expect accuracy: %.2f, get accuracy: %.2f" %
          (batch3_acc, accuracy_manager.eval()))
    acc = batch3_acc
    tools.compare(acc, accuracy_manager.eval())


def test_accuracy2():
    """
    test accuacy with different batchsize
    :return:
    """
    import paddle.fluid as fluid
    batch1_size = 128
    accuracy_manager = fluid.metrics.Accuracy()
    # 假设第一个batch的准确率为0.9
    batch1_acc = 0.9
    accuracy_manager.update(value=batch1_acc, weight=batch1_size)
    print("expect accuracy: %.2f, get accuracy: %.2f" %
          (batch1_acc, accuracy_manager.eval()))
    acc = batch1_acc
    tools.compare(acc, accuracy_manager.eval())
    # 假设第二个batch的准确率为0.8
    batch2_size = 64
    batch2_acc = 0.8
    accuracy_manager.update(value=batch2_acc, weight=batch2_size)
    print("expect accuracy: %.2f, get accuracy: %.2f" %
          ((batch1_acc * batch1_size + batch2_acc * batch2_size) /
           (batch1_size + batch2_size), accuracy_manager.eval()))
    acc = (batch1_acc * batch1_size + batch2_acc * batch2_size) / (
        batch1_size + batch2_size)
    tools.compare(acc, accuracy_manager.eval())
    # 假设第三个batch的准确率为0.4
    batch3_size = 32
    batch3_acc = 0.4
    accuracy_manager.update(value=batch3_acc, weight=batch3_size)
    print("expect accuracy: %.2f, get accuracy: %.2f" %
          ((batch1_acc * batch1_size + batch2_acc * batch2_size + batch3_acc *
            batch3_size) / (batch1_size + batch2_size + batch3_size),
           accuracy_manager.eval()))
    acc = (batch1_acc * batch1_size + batch2_acc * batch2_size + batch3_acc * batch3_size) / \
          (batch1_size + batch2_size + batch3_size)
    tools.compare(acc, accuracy_manager.eval())


def test_Auc():
    """
    test Auc
    :return:
    """
    # 初始化auc度量
    auc_metric = fluid.metrics.Auc("ROC")
    # 假设batch_size为128
    batch_num = 1000
    batch_size = 128
    for batch_id in range(batch_num):
        class0_preds = np.random.random(size=(batch_size, 1))
        class1_preds = 1 - class0_preds

        preds = np.concatenate((class0_preds, class1_preds), axis=1)
        labels = np.random.randint(2, size=(batch_size, 1))
        auc_metric.update(preds=preds, labels=labels)

        # 应为一个接近0.5的值，因为preds是随机指定的
        if batch_id % 200 == 0:
            print("auc for iteration %d is %.2f" %
                  (batch_id, auc_metric.eval()))
    tools.compare(0.5, auc_metric.eval(), delta=1e-2)


def test_Auc1():
    """
    test Auc with curve=PR name=A
    :return:
    """
    # 初始化auc度量
    auc_metric = fluid.metrics.Auc(name="A", curve="PR")

    # 假设batch_size为128
    batch_num = 1000
    batch_size = 128
    for batch_id in range(batch_num):
        class0_preds = np.random.random(size=(batch_size, 1))
        class1_preds = 1 - class0_preds

        preds = np.concatenate((class0_preds, class1_preds), axis=1)
        labels = np.random.randint(2, size=(batch_size, 1))
        auc_metric.update(preds=preds, labels=labels)

        # 应为一个接近0.5的值，因为preds是随机指定的
        if batch_id % 200 == 0:
            print("auc for iteration %d is %.2f" %
                  (batch_id, auc_metric.eval()))
    tools.compare(0.5, auc_metric.eval(), delta=1e-2)


def test_Auc2():
    """
    test Auc with curve=PR name=A num_thresholds=8888
    :return:
    """
    # 初始化auc度量
    auc_metric = fluid.metrics.Auc(name="A", curve="PR", num_thresholds=8888)

    # 假设batch_size为128
    batch_num = 1000
    batch_size = 128
    for batch_id in range(batch_num):
        class0_preds = np.random.random(size=(batch_size, 1))
        class1_preds = 1 - class0_preds

        preds = np.concatenate((class0_preds, class1_preds), axis=1)
        labels = np.random.randint(2, size=(batch_size, 1))
        auc_metric.update(preds=preds, labels=labels)

        # 应为一个接近0.5的值，因为preds是随机指定的
        if batch_id % 200 == 0:
            print("auc for iteration %d is %.2f" %
                  (batch_id, auc_metric.eval()))
    tools.compare(0.5, auc_metric.eval(), delta=1e-2)


def test_ChunkEvaluator():
    """
    test ChunkEvaluator
    :return:
    """
    # 初始化chunck-level的评价管理。
    metric = fluid.metrics.ChunkEvaluator()
    # 假设模型预测10个chuncks，其中8个为正确，且真值有9个chuncks。
    num_infer_chunks = 10
    num_label_chunks = 9
    num_correct_chunks = 8
    metric.update(num_infer_chunks, num_label_chunks, num_correct_chunks)
    numpy_precision, numpy_recall, numpy_f1 = metric.eval()
    print("precision: %.2f, recall: %.2f, f1: %.2f" %
          (numpy_precision, numpy_recall, numpy_f1))
    tools.compare(0.80, numpy_precision, delta=1e-2)
    tools.compare(0.89, numpy_recall, delta=1e-2)
    tools.compare(0.84, numpy_f1, delta=1e-2)
    # 下一个batch，完美地预测了3个正确的chuncks。
    num_infer_chunks = 3
    num_label_chunks = 3
    num_correct_chunks = 3
    metric.update(num_infer_chunks, num_label_chunks, num_correct_chunks)
    numpy_precision, numpy_recall, numpy_f1 = metric.eval()
    print("precision: %.2f, recall: %.2f, f1: %.2f" %
          (numpy_precision, numpy_recall, numpy_f1))
    tools.compare(0.85, numpy_precision, delta=1e-2)
    tools.compare(0.92, numpy_recall, delta=1e-2)
    tools.compare(0.88, numpy_f1, delta=1e-2)


def test_ChunkEvaluator1():
    """
    test ChunkEvaluator with name=chunk
    :return:
    """
    # 初始化chunck-level的评价管理。
    metric = fluid.metrics.ChunkEvaluator(name="chunk")
    # 假设模型预测10个chuncks，其中8个为正确，且真值有9个chuncks。
    num_infer_chunks = 10
    num_label_chunks = 9
    num_correct_chunks = 8
    metric.update(num_infer_chunks, num_label_chunks, num_correct_chunks)
    numpy_precision, numpy_recall, numpy_f1 = metric.eval()
    print("precision: %.2f, recall: %.2f, f1: %.2f" %
          (numpy_precision, numpy_recall, numpy_f1))
    tools.compare(0.80, numpy_precision, delta=1e-2)
    tools.compare(0.89, numpy_recall, delta=1e-2)
    tools.compare(0.84, numpy_f1, delta=1e-2)
    # 下一个batch，完美地预测了3个正确的chuncks。
    num_infer_chunks = 3
    num_label_chunks = 3
    num_correct_chunks = 3
    metric.update(num_infer_chunks, num_label_chunks, num_correct_chunks)
    numpy_precision, numpy_recall, numpy_f1 = metric.eval()
    print("precision: %.2f, recall: %.2f, f1: %.2f" %
          (numpy_precision, numpy_recall, numpy_f1))
    tools.compare(0.85, numpy_precision, delta=1e-2)
    tools.compare(0.92, numpy_recall, delta=1e-2)
    tools.compare(0.88, numpy_f1, delta=1e-2)


def test_CompositeMetric():
    """
    test CompositeMetric
    :return:
    """
    preds = [[0.1], [0.7], [0.8], [0.9], [0.2], [0.2], [0.3], [0.5], [0.8],
             [0.6]]
    labels = [[0], [1], [1], [1], [1], [0], [0], [0], [0], [0]]
    preds = np.array(preds)
    labels = np.array(labels)

    comp = fluid.metrics.CompositeMetric()
    precision = fluid.metrics.Precision()
    recall = fluid.metrics.Recall()
    comp.add_metric(precision)
    comp.add_metric(recall)

    comp.update(preds=preds, labels=labels)
    numpy_precision, numpy_recall = comp.eval()
    print("expect precision: %.2f, got %.2f" % (3. / 5, numpy_precision))
    print("expect recall: %.2f, got %.2f" % (3. / 4, numpy_recall))
    tools.compare(3. / 5, numpy_precision, delta=1e-2)
    tools.compare(3. / 4, numpy_recall, delta=1e-2)


def test_CompositeMetric1():
    """
    test CompositeMetric name="CompositeMetric"
    :return:
    """
    preds = [[0.1], [0.7], [0.8], [0.9], [0.2], [0.2], [0.3], [0.5], [0.8],
             [0.6]]
    labels = [[0], [1], [1], [1], [1], [0], [0], [0], [0], [0]]
    preds = np.array(preds)
    labels = np.array(labels)

    comp = fluid.metrics.CompositeMetric(name="CompositeMetric")
    precision = fluid.metrics.Precision()
    recall = fluid.metrics.Recall()
    comp.add_metric(precision)
    comp.add_metric(recall)

    comp.update(preds=preds, labels=labels)
    numpy_precision, numpy_recall = comp.eval()
    print("expect precision: %.2f, got %.2f" % (3. / 5, numpy_precision))
    print("expect recall: %.2f, got %.2f" % (3. / 4, numpy_recall))
    tools.compare(3. / 5, numpy_precision, delta=1e-2)
    tools.compare(3. / 4, numpy_recall, delta=1e-2)


def test_EditDistance():
    """
    test EditDistance
    :return:
    """
    batch_size = 128
    np.random.seed(33)
    # 初始化编辑距离管理器
    distance_evaluator = fluid.metrics.EditDistance("EditDistance")
    # 生成128个序列对间的编辑距离，此处的最大距离是10
    edit_distances_batch0 = np.random.randint(
        low=0, high=10, size=(batch_size, 1))
    seq_num_batch0 = batch_size

    distance_evaluator.update(edit_distances_batch0, seq_num_batch0)
    avg_distance, wrong_instance_ratio = distance_evaluator.eval()
    print(
        "the average edit distance for batch0 is %.2f and the wrong instance ratio is %.2f "
        % (avg_distance, wrong_instance_ratio))
    tools.compare(avg_distance, 4.53, 1e-2)
    tools.compare(wrong_instance_ratio, 0.90, 1e-2)

    edit_distances_batch1 = np.random.randint(
        low=0, high=10, size=(batch_size, 1))
    seq_num_batch1 = batch_size

    distance_evaluator.update(edit_distances_batch1, seq_num_batch1)
    avg_distance, wrong_instance_ratio = distance_evaluator.eval()
    print(
        "the average edit distance for batch0 and batch1 is %.2f and the wrong instance ratio is %.2f "
        % (avg_distance, wrong_instance_ratio))
    tools.compare(avg_distance, 4.50, 1e-2)
    tools.compare(wrong_instance_ratio, 0.88, 1e-2)


def test_EditDistance1():
    """
    test EditDistance reset
    :return:
    """
    batch_size = 128
    np.random.seed(33)
    # 初始化编辑距离管理器
    distance_evaluator = fluid.metrics.EditDistance("EditDistance")
    # 生成128个序列对间的编辑距离，此处的最大距离是10
    edit_distances_batch0 = np.random.randint(
        low=0, high=10, size=(batch_size, 1))
    seq_num_batch0 = batch_size

    distance_evaluator.update(edit_distances_batch0, seq_num_batch0)
    avg_distance, wrong_instance_ratio = distance_evaluator.eval()
    print(
        "the average edit distance for batch0 is %.2f and the wrong instance ratio is %.2f "
        % (avg_distance, wrong_instance_ratio))
    tools.compare(avg_distance, 4.53, 1e-2)
    tools.compare(wrong_instance_ratio, 0.90, 1e-2)
    try:
        distance_evaluator.reset()
        avg_distance, wrong_instance_ratio = distance_evaluator.eval()
        print(
            "the average edit distance for batch0 is %.2f and the wrong instance ratio is %.2f "
            % (avg_distance, wrong_instance_ratio))
        assert False
    except ValueError:
        assert True


def test_Precision():
    """
    test Precision
    :return:
    """
    metric = fluid.metrics.Precision()
    # 生成预测值和标签
    preds = [[0.1], [0.7], [0.8], [0.9], [0.2], [0.2], [0.3], [0.5], [0.8],
             [0.6]]
    labels = [[0], [1], [1], [1], [1], [0], [0], [0], [0], [0]]
    preds = np.array(preds)
    labels = np.array(labels)
    metric.update(preds=preds, labels=labels)
    precision = metric.eval()
    print("expected precision: %.2f and got %.2f" % (3.0 / 5.0, precision))
    tools.compare(precision, 3.0 / 5.0)


def test_Precision1():
    """
    test Precision with name = Precision
    :return:
    """
    metric = fluid.metrics.Precision("Precision")
    # 生成预测值和标签
    preds = [[0.1], [0.7], [0.8], [0.9], [0.2], [0.2], [0.3], [0.5], [0.8],
             [0.6]]
    labels = [[0], [1], [1], [1], [1], [0], [0], [0], [0], [0]]
    preds = np.array(preds)
    labels = np.array(labels)
    metric.update(preds=preds, labels=labels)
    precision = metric.eval()
    print("expected precision: %.2f and got %.2f" % (3.0 / 5.0, precision))
    tools.compare(precision, 3.0 / 5.0)


def test_Recall():
    """
    test Recall
    :return:
    """
    metric = fluid.metrics.Recall()
    # 生成预测值和标签
    preds = [[0.1], [0.7], [0.8], [0.9], [0.2], [0.2], [0.3], [0.5], [0.8],
             [0.6]]
    labels = [[0], [1], [1], [1], [1], [0], [0], [0], [0], [0]]
    preds = np.array(preds)
    labels = np.array(labels)
    metric.update(preds=preds, labels=labels)
    recall = metric.eval()
    print("expected recall: %.2f and got %.2f" % (3.0 / 4.0, recall))
    tools.compare(recall, 3.0 / 4.0)


def test_Recall1():
    """
    test Recall with name=Recall
    :return:
    """
    metric = fluid.metrics.Recall("Recall")
    # 生成预测值和标签
    preds = [[0.1], [0.7], [0.8], [0.9], [0.2], [0.2], [0.3], [0.5], [0.8],
             [0.6]]
    labels = [[0], [1], [1], [1], [1], [0], [0], [0], [0], [0]]
    preds = np.array(preds)
    labels = np.array(labels)
    metric.update(preds=preds, labels=labels)
    recall = metric.eval()
    print("expected recall: %.2f and got %.2f" % (3.0 / 4.0, recall))
    tools.compare(recall, 3.0 / 4.0)


def test_MetricBase():
    """
    test MetricBase
    :return:
    """
    # 生成预测值和标签
    preds = [[0.1], [0.7], [0.8], [0.9], [0.2], [0.2], [0.3], [0.5], [0.8],
             [0.6]]
    labels = [[0], [1], [1], [1], [1], [0], [0], [0], [0], [0]]
    try:
        metric = fluid.metrics.MetricBase("Metric")
        metric.update(preds, labels)
        res = metric.eval()
        print(res)
        assert False
    except NotImplementedError:
        assert True


def test_DetectionMAP():
    """
    test DetectionMAP
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            detect_res = fluid.layers.data(
                name='detect_res',
                shape=[10, 6],
                append_batch_size=False,
                dtype='float32')
            label = fluid.layers.data(
                name='label',
                shape=[10, 1],
                append_batch_size=False,
                dtype='float32')
            box = fluid.layers.data(
                name='bbox',
                shape=[10, 4],
                append_batch_size=False,
                dtype='float32')
            map_eval = fluid.metrics.DetectionMAP(
                detect_res, label, box, class_num=21)
            cur_map, accm_map = map_eval.get_map_var()
            assert cur_map is not None
            assert accm_map is not None


def test_DetectionMAP1():
    """
    test DetectionMAP background_label=-1
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            detect_res = fluid.layers.data(
                name='detect_res',
                shape=[10, 6],
                append_batch_size=False,
                dtype='float32')
            label = fluid.layers.data(
                name='label',
                shape=[10, 1],
                append_batch_size=False,
                dtype='float32')
            box = fluid.layers.data(
                name='bbox',
                shape=[10, 4],
                append_batch_size=False,
                dtype='float32')
            map_eval = fluid.metrics.DetectionMAP(
                detect_res, label, box, class_num=21, background_label=-1)
            cur_map, accm_map = map_eval.get_map_var()
            assert cur_map is not None
            assert accm_map is not None


def test_DetectionMAP2():
    """
    test DetectionMAP overlap_threshold=0.8
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            detect_res = fluid.layers.data(
                name='detect_res',
                shape=[10, 6],
                append_batch_size=False,
                dtype='float32')
            label = fluid.layers.data(
                name='label',
                shape=[10, 1],
                append_batch_size=False,
                dtype='float32')
            box = fluid.layers.data(
                name='bbox',
                shape=[10, 4],
                append_batch_size=False,
                dtype='float32')
            map_eval = fluid.metrics.DetectionMAP(
                detect_res,
                label,
                box,
                class_num=21,
                background_label=-1,
                overlap_threshold=0.8)
            cur_map, accm_map = map_eval.get_map_var()
            assert cur_map is not None
            assert accm_map is not None


def test_DetectionMAP3():
    """
    test DetectionMAP overlap_threshold=0.8 ap_version='11point'
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            detect_res = fluid.layers.data(
                name='detect_res',
                shape=[10, 6],
                append_batch_size=False,
                dtype='float32')
            label = fluid.layers.data(
                name='label',
                shape=[10, 1],
                append_batch_size=False,
                dtype='float32')
            box = fluid.layers.data(
                name='bbox',
                shape=[10, 4],
                append_batch_size=False,
                dtype='float32')
            map_eval = fluid.metrics.DetectionMAP(
                detect_res,
                label,
                box,
                class_num=21,
                background_label=-1,
                overlap_threshold=0.8,
                ap_version='11point')
            cur_map, accm_map = map_eval.get_map_var()
            assert cur_map is not None
            assert accm_map is not None
