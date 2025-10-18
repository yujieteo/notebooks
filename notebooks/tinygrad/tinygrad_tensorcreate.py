from tinygrad import dtypes
from tinygrad import Tensor

t3 = Tensor([1, 2, 3, 4, 5], dtype=dtypes.int32)

t4 = Tensor([1, 2, 3, 4, 5])
t5 = (t4 + 1) * 2
t6 = (t5 * t4).relu().log_softmax()

print(t6.numpy())
# [-56. -48. -36. -20.   0.]