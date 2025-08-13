#!/usr/bin/env python

"""Wrapper script for py_compile which reformats SyntaxError,
IndentationError and TabError into a format more palatable to BBEdit and
TextWrangler's Python error parsers."""

from __future__ import print_function

import os
import sys
import py_compile

from pprint import pprint

__all__ = ['main']


def print_reformatted_error(exc_type_name, exc_value):
	"""Print out compile exceptions in the same friendly format that they
	are dumped at runtime when running python with compile errors.

	We started included SyntaxError in the list of exceptions to be printed
	for Python 2.6 since it stopped being pretty-printed by default. 
	
	This is unecessary, and causes a syntax error, for Python 3.	
	"""
	if isinstance(exc_value, SyntaxError):
		msg = exc_value.msg
		file = exc_value.filename
		line = exc_value.lineno
		col = exc_value.offset
		ctx = exc_value.text
	else:
		msg = exc_value[0]
		file = exc_value[1][0]
		line = exc_value[1][1]
		col = exc_value[1][2]
		ctx = exc_value[1][3]
	if ctx.endswith('\r') or ctx.endswith('\n'):
		ctx = ctx[:-1]
	print('''  File "%s", line %d''' % (file, line), file=sys.stdout)
	print(ctx, file=sys.stdout)
	if col is not None:
		print(' ' * col + '^', file=sys.stdout)
	print(exc_type_name + ": " + msg, file=sys.stdout)


def check_syntax(source_file):
	"""Check syntax on file"""
	try:
		py_compile.compile(source_file, doraise=True)
	except py_compile.PyCompileError as ex:
		(c, v, t) = sys.exc_info()
		exc_names = [SyntaxError.__name__, IndentationError.__name__, TabError.__name__]
		if v.exc_type_name in exc_names:
			print_reformatted_error(v.exc_type_name, v.exc_value)
		else:
			print(v, end=' ', file=sys.stdout)
		return 1
	return 0


def fix_path():
	self_path = sys.argv[0]
	self_parent = os.path.dirname(self_path)
	if self_parent not in sys.path:
		sys.path.insert(0, self_parent)


def main():
	"""Check syntax on the file passed in sys.argv[1]"""
	source_file = sys.argv[1]
	fix_path()
	rc = check_syntax(source_file)
	sys.exit(rc)

if __name__ == "__main__":
	main()
