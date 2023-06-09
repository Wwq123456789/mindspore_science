mindsponge.function.calc_angle_between_vectors
==============================================

.. py:function:: mindsponge.function.calc_angle_between_vectors(vector1, vector2)

    计算两个向量之间的角。对于向量 :math:`\vec {V_1} = (x_1, x_2, x_3, ..., x_n)` 和向量 :math:`\vec {V_2} = (y_1, y_2, y_3, ..., y_n)` ，两向量间夹角计算公式为：

    .. math::

        \theta = \arccos {\frac{|x_1y_1 + x_2y_2 + \cdots + x_ny_n|}{\sqrt{x_1^2 + x_2^2 + \cdots + x_n^2}\sqrt{y_1^2 + y_2^2 + \cdots + y_n^2}}}

    参数：
        - **vector1** (Tensor) - 向量1，shape为 :math:`(..., D)` ，数据类型为float。
        - **vector2** (Tensor) - 向量2，shape为 :math:`(..., D)` ，数据类型为float。

    输出：
        Tensor。计算所得角。shape为 :math:`(..., 1)` ，数据类型为float。

    符号：
        - **D** - 模拟系统的维度, 一般为3。