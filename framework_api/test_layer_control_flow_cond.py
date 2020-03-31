# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""test layer control flow."""
import paddle.fluid as fluid
import numpy as np

def test_base_cond():
    """
    a=0.23
    b=0.24
    if a<b
        a+b=0.47
    else
        b*b 
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(main_program, startup_program):
        a = fluid.layers.fill_constant(shape=[1], dtype='float32', value=0.23)
        b = fluid.layers.fill_constant(shape=[1], dtype='float32', value=0.24)
        out = fluid.layers.cond(a < b, lambda: a + b, lambda: a * b)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        ret = exe.run(main_program, fetch_list=[out])
        assert ret == [0.47]


def test_cond_with_elementwise():
    """
    use elementwise_add and elementwise_mul
    a=1.23
    b=1.24
    if a<b
        a+b=2.47
    else
        b*b 
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(main_program, startup_program):
        a = fluid.layers.fill_constant(shape=[1], dtype='float32', value=1.23)
        b = fluid.layers.fill_constant(shape=[1], dtype='float32', value=1.24)
        out = fluid.layers.cond(a < b, lambda: fluid.layers.elementwise_add(a ,b), lambda: fluid.layers.elementwise_mul(a, b))
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        ret = exe.run(main_program, fetch_list=[out])
        assert ret == [2.47]


def test_cond_with_stop_gradient():
    """
    use stop gradient
    a=1.23
    b=1.24
    if a<b
        a+b=2.47
    else
        b*b 
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(main_program, startup_program):
        a = fluid.layers.fill_constant(shape=[1], dtype='float32', value=1.23)
        b = fluid.layers.fill_constant(shape=[1], dtype='float32', value=1.24)
        out = fluid.layers.cond(a < b, lambda: fluid.layers.elementwise_add(a ,b), lambda: fluid.layers.elementwise_mul(a, b))
        out.stop_gradient = False
        fluid.backward.append_backward(out)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        ret = exe.run(main_program, fetch_list=[out])
        assert ret == [2.47]


def test_cond_nest():
    """
    嵌套条件表达式
    a=1.23
    b=1.24
    if a<b
        if a-b<-1
            a+b=2.47
        else
            a*b
    else
        if a=b
            a-b
        else
            a**b
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(main_program, startup_program):
        a = fluid.layers.fill_constant(shape=[1], dtype='float32', value=1.23)
        b = fluid.layers.fill_constant(shape=[1], dtype='float32', value=1.24)
        out = fluid.layers.cond(fluid.layers.less_than(a, b), 
                                lambda: fluid.layers.cond(a - b < -1, 
                                                lambda: fluid.layers.elementwise_add(a, b), 
                                                lambda: fluid.layers.elementwise_mul(a, b)),
                                lambda: fluid.layers.cond(fluid.layers.equal(a, b), 
                                                lambda: fluid.layers.elementwise_sub(a ,b), 
                                                lambda: fluid.layers.elementwise_pow(a, b))
                                )
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        ret = exe.run(main_program, fetch_list=[out])
        assert ret == [1.5252]


def test_cond_nest_with_stop_gradient():
    """
    嵌套条件表达式,附带stop_gradient
    a=1.23
    b=1.24
    if a<b
        if a-b<-1
            a+b=2.47
        else
            a*b
    else
        if a=b
            a-b
        else
            a**b
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(main_program, startup_program):
        a = fluid.layers.fill_constant(shape=[1], dtype='float32', value=1.23)
        b = fluid.layers.fill_constant(shape=[1], dtype='float32', value=1.24)
        out = fluid.layers.cond(fluid.layers.less_than(a, b), 
                                lambda: fluid.layers.cond(a - b < -1, 
                                                lambda: fluid.layers.elementwise_add(a, b), 
                                                lambda: fluid.layers.elementwise_mul(a, b)),
                                lambda: fluid.layers.cond(fluid.layers.equal(a, b), 
                                                lambda: fluid.layers.elementwise_sub(a ,b), 
                                                lambda: fluid.layers.elementwise_pow(a, b))
                                )
        out.stop_gradient = False
        fluid.backward.append_backward(out)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        ret = exe.run(main_program, fetch_list=[out])
        assert ret == [1.5252]


def test_cond_linear():
    """
    linear
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    startup_program.random_seed = 1
    main_program.random_seed = 1
    with fluid.program_guard(main_program, startup_program):
        outputs = np.asarray([(1,2,3,4),(5,6,7,8),(9,10,11,12),(13,14,15,16)])
        print(outputs)
        res = []
        for i in range(4):
            # 假设方程式为 y=4a+6b+7c+2d
            y = 4*outputs[i][0]+6*outputs[i][1]+7*outputs[i][2]+2*outputs[i][3]
            res.append([y])
        # 定义数据
        train_data=np.array(outputs).astype('float32')
        y_true = np.array(res).astype('float32')
        #定义网络
        x = fluid.layers.data(name="x",shape=[4],dtype='float32')
        y = fluid.layers.data(name="y",shape=[1],dtype='float32')
        y_predict = fluid.layers.fc(input=x,size=1,act=None)
        #定义损失函数
        cost = fluid.layers.square_error_cost(input=y_predict,label=y)
        avg_cost = fluid.layers.mean(cost)
        #定义优化方法
        def sgd_optimizer(learning_rate):
            return avg_cost + learning_rate

        avg_cost = fluid.layers.cond(fluid.layers.reduce_sum(x) < 40, lambda: sgd_optimizer(0.05), lambda: sgd_optimizer(0.08))
        #参数初始化
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        actual_cost = []
        for i in range(4):
            outs = exe.run(
                feed={'x':train_data[i:i+1],'y':y_true[i:i+1]},
                fetch_list=[x, fluid.layers.reduce_sum(x), y_predict.name,avg_cost.name])
            print(outs)
            actual_cost.append(outs[3])
        assert actual_cost == [1919.859, 15255.681, 41295.094, 80038.03]
    
def test_base_cond_in_clone():
    """
    a=0.23
    b=0.24
    if a<b
        a+b=0.47
    else
        b*b 
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(main_program, startup_program):
        a = fluid.layers.fill_constant(shape=[1], dtype='float32', value=0.23)
        b = fluid.layers.fill_constant(shape=[1], dtype='float32', value=0.24)
        out = fluid.layers.cond(a < b, lambda: a + b, lambda: a * b)
        test_program = main_program.clone(for_test=True)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        ret = exe.run(test_program, fetch_list=[out])
        assert ret == [0.47]

