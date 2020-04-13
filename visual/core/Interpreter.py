from typing import Tuple
import sys

import pandas as pd


class Interpreter:

	def __init__(self, pandas_data_object: pd.DataFrame):
		self.current_object = pandas_data_object
		self._root_copy = pandas_data_object.copy(deep=True)
		pass

	def eval_code(self, code_str) -> Tuple[bool, str, pd.DataFrame()]:
		data = self.current_object
		try:
			result = None
			result = eval(code_str, __globals={}, __locals={'data': data})
		except Exception as exc:
			print("Error occured", sys.exc_info()[0])
			return False, str(sys.exc_info()[0]), self.current_object
		self.current_object = self._root_copy
		self._root_copy = self._root_copy.copy(deep=True)
		return True, "", result
