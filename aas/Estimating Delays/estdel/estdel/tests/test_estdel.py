import unittest
from estdel.estdel import _DelayPredict
import numpy as np

class test_DelayPredict(unittest.TestCase):

	def test_init(self):

		data = np.exp(-2j*np.pi*np.arange(1024)*0.01).reshape(-1,1024)

		delayPredict = _DelayPredict(data)

		self.assertEqual(delayPredict._n_freqs, 1024)
		np.testing.assert_array_equal(delayPredict._data, data)

		data_2 = np.exp(-2j*np.pi*np.arange(1024 + 1)*0.01).reshape(-1,1024 + 1)

		self.assertRaises(AssertionError, _DelayPredict, data_2)




if __name__ == '__main__':
	unittest.main()
