import numpy
import pprint
import matplotlib.pyplot as plt

zeros = numpy.zeros([3, 3])

pprint.pprint(zeros)

zeros[0][0] = 1
zeros[0][1] = 1.1
zeros[0][2] = 1

pprint.pprint(zeros)

plt.imshow(zeros, interpolation="nearest")
plt.show()
