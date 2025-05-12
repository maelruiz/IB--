#!/usr/bin/env python3
"""
IB++ Language Interpreter
------------------------
A custom interpreter for a BASIC-like language, supporting variables, functions, lists, collections, control flow, and more.

This file contains the full implementation of the lexer, parser, AST nodes, runtime values, built-in functions, and interpreter logic.

Author: (your name here)
"""
 
#######################################
# IMPORTS
#######################################

from strings_with_arrows import *

import string
import os
import math
import random as pyrandom

#######################################
# CONSTANTS
#######################################

DIGITS = '0123456789'
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS

#######################################
# ERRORS
#######################################

class Error:
  """Base class for all errors in the language."""
  def __init__(self, pos_start, pos_end, error_name, details):
    """
    Initialize an error.
    Args:
        pos_start (Position): Start position of the error.
        pos_end (Position): End position of the error.
        error_name (str): Name of the error type.
        details (str): Error details.
    """
    self.pos_start = pos_start
    self.pos_end = pos_end
    self.error_name = error_name
    self.details = details

  def as_string(self):
    """Return a string representation of the error with context."""
    result  = f'{self.error_name}: {self.details}\n'
    result += f'File {self.pos_start.fn}, line {self.pos_start.ln + 1}, column {self.pos_start.col + 1}'
    result += '\n\nCode context:\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
    return result

class IllegalCharError(Error):
  """Error for illegal characters in the source code."""
  def __init__(self, pos_start, pos_end, details):
    super().__init__(pos_start, pos_end, 'Illegal Character', details)

class ExpectedCharError(Error):
  """Error for missing expected characters in the source code."""
  def __init__(self, pos_start, pos_end, details):
    super().__init__(pos_start, pos_end, 'Expected Character', details)

class InvalidSyntaxError(Error):
  """Error for invalid syntax in the source code."""
  def __init__(self, pos_start, pos_end, details=''):
    super().__init__(pos_start, pos_end, 'Invalid Syntax', details)

class RTError(Error):
  """Runtime error during program execution."""
  def __init__(self, pos_start, pos_end, details, context):
    super().__init__(pos_start, pos_end, 'Runtime Error', details)
    self.context = context

  def as_string(self):
    """Return a string representation of the runtime error with traceback."""
    result  = self.generate_traceback()
    result += f'{self.error_name}: {self.details}'
    result += '\n\nCode context:\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
    return result

  def generate_traceback(self):
    """Generate a traceback string for the error context."""
    result = '\nTraceback (most recent call last):\n'
    pos = self.pos_start
    ctx = self.context

    frames = []
    while ctx:
      frames.append((pos, ctx))
      pos = ctx.parent_entry_pos
      ctx = ctx.parent

    # Print traceback from earliest to most recent
    for i, (pos, ctx) in enumerate(reversed(frames)):
      result += f'  [{i}] File {pos.fn}, line {str(pos.ln + 1)}, column {pos.col + 1}, in {ctx.display_name}\n'
      
      # Add context lines (show the actual code)
      if pos.ftxt:
        lines = pos.ftxt.split('\n')
        if 0 <= pos.ln < len(lines):
          line = lines[pos.ln]
          result += f'      {line}\n'
          result += f'      {" " * pos.col}^\n'
    
    return result

#######################################
# POSITION
#######################################

class Position:
  """Represents a position in the source code."""
  def __init__(self, idx, ln, col, fn, ftxt):
    """
    Args:
        idx (int): Character index in the file.
        ln (int): Line number.
        col (int): Column number.
        fn (str): Filename.
        ftxt (str): Full file text.
    """
    self.idx = idx
    self.ln = ln
    self.col = col
    self.fn = fn
    self.ftxt = ftxt

  def advance(self, current_char=None):
    """Advance the position by one character."""
    self.idx += 1
    self.col += 1

    if current_char == '\n':
      self.ln += 1
      self.col = 0

    return self

  def copy(self):
    """Return a copy of this position."""
    return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)

#######################################
# TOKENS
#######################################

TT_INT        = 'INT'
TT_FLOAT      = 'FLOAT'
TT_STRING     = 'STRING'
TT_IDENTIFIER = 'IDENTIFIER'
TT_KEYWORD    = 'KEYWORD'
TT_PLUS       = 'PLUS'
TT_MINUS      = 'MINUS'
TT_MUL        = 'MUL'
TT_DIV        = 'DIV'
TT_POW        = 'POW'
TT_EQ         = 'EQ'
TT_LPAREN     = 'LPAREN'
TT_RPAREN     = 'RPAREN'
TT_LSQUARE    = 'LSQUARE'
TT_RSQUARE    = 'RSQUARE'
TT_EE         = 'EE'
TT_NE         = 'NE'
TT_LT         = 'LT'
TT_GT         = 'GT'
TT_LTE        = 'LTE'
TT_GTE        = 'GTE'
TT_COMMA      = 'COMMA'
TT_ARROW      = 'ARROW'
TT_NEWLINE    = 'NEWLINE'
TT_EOF        = 'EOF'
TT_MOD        = 'MOD'
TT_DOT        = 'DOT'
TT_DIV_INT    = 'DIV_INT'
TT_MOD_INT    = 'MOD_INT'

KEYWORDS = [
  'and',
  'or',
  'not',
  'if',
  'elif',
  'else',
  'to',
  'step',
  'while',
  'method', 
  'then',
  'end',
  'return',
  'continue',
  'break',
  'loop',
  'from',
  'output',
  'input',
  'new',
  'div',
  'mod',
]

class Token:
  def __init__(self, type_, value=None, pos_start=None, pos_end=None):
    self.type = type_
    self.value = value

    if pos_start:
      self.pos_start = pos_start.copy()
      self.pos_end = pos_start.copy()
      self.pos_end.advance()

    if pos_end:
      self.pos_end = pos_end.copy()

  def matches(self, type_, value):
    return self.type == type_ and self.value == value
  
  def __repr__(self):
    if self.value: return f'{self.type}:{self.value}'
    return f'{self.type}'

class Lexer:
  def __init__(self, fn, text):
    self.fn = fn
    self.text = text
    self.pos = Position(-1, 0, -1, fn, text)
    self.current_char = None
    self.advance()
  
  def advance(self):
    self.pos.advance(self.current_char)
    self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

  def make_tokens(self):
    tokens = []

    while self.current_char != None:
      if self.current_char in ' \t':
        self.advance()
      elif self.current_char == '/' and self.peek() == '/':
        self.advance()
        self.advance()  
        self.skip_comment()
      elif self.current_char in ';\n':
        tokens.append(Token(TT_NEWLINE, pos_start=self.pos))
        self.advance()
      elif self.current_char in DIGITS:
        tokens.append(self.make_number())
      elif self.current_char in LETTERS:
        tokens.append(self.make_identifier())
      elif self.current_char == '"':
        tokens.append(self.make_string())
      elif self.current_char == '+':
        tokens.append(Token(TT_PLUS, pos_start=self.pos))
        self.advance()
      elif self.current_char == '-':
        tokens.append(self.make_minus_or_arrow())
      elif self.current_char == '*':
        tokens.append(Token(TT_MUL, pos_start=self.pos))
        self.advance()
      elif self.current_char == '/':
        tokens.append(Token(TT_DIV, pos_start=self.pos))
        self.advance()
      elif self.current_char == '^':
        tokens.append(Token(TT_POW, pos_start=self.pos))
        self.advance()
      elif self.current_char == '(':
        tokens.append(Token(TT_LPAREN, pos_start=self.pos))
        self.advance()
      elif self.current_char == '%':
        tokens.append(Token(TT_MOD, pos_start=self.pos))
        self.advance()
      elif self.current_char == ')':
        tokens.append(Token(TT_RPAREN, pos_start=self.pos))
        self.advance()
      elif self.current_char == '[':
        tokens.append(Token(TT_LSQUARE, pos_start=self.pos))
        self.advance()
      elif self.current_char == ']':
        tokens.append(Token(TT_RSQUARE, pos_start=self.pos))
        self.advance()
      elif self.current_char == '.': 
        tokens.append(Token(TT_DOT, pos_start=self.pos))
        self.advance()
      elif self.current_char == '!':
        token, error = self.make_not_equals()
        if error: return [], error
        tokens.append(token)
      elif self.current_char == '=':
        tokens.append(self.make_equals())
      elif self.current_char == '<':
        tokens.append(self.make_less_than())
      elif self.current_char == '>':
        tokens.append(self.make_greater_than())
      elif self.current_char == ',':
        tokens.append(Token(TT_COMMA, pos_start=self.pos))
        self.advance()
      else:
        pos_start = self.pos.copy()
        char = self.current_char
        self.advance()
        return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")

    tokens.append(Token(TT_EOF, pos_start=self.pos))
    return tokens, None

  def peek(self):
    peek_idx = self.pos.idx + 1
    if peek_idx < len(self.text):
      return self.text[peek_idx]
    return None

  def make_number(self):
    num_str = ''
    dot_count = 0
    pos_start = self.pos.copy()

    while self.current_char != None and self.current_char in DIGITS + '.':
      if self.current_char == '.':
        if dot_count == 1: break
        dot_count += 1
      num_str += self.current_char
      self.advance()

    if dot_count == 0:
      return Token(TT_INT, int(num_str), pos_start, self.pos)
    else:
      return Token(TT_FLOAT, float(num_str), pos_start, self.pos)

  def make_string(self):
    string = ''
    pos_start = self.pos.copy()
    escape_character = False
    self.advance()

    escape_characters = {
      'n': '\n',
      't': '\t'
    }

    while self.current_char != None and (self.current_char != '"' or escape_character):
      if escape_character:
        string += escape_characters.get(self.current_char, self.current_char)
      else:
        if self.current_char == '\\':
          escape_character = True
        else:
          string += self.current_char
      self.advance()
      escape_character = False
    
    self.advance()
    return Token(TT_STRING, string, pos_start, self.pos)

  def make_identifier(self):
    id_str = ''
    pos_start = self.pos.copy()

    while self.current_char != None and self.current_char in LETTERS_DIGITS + '_':
        id_str += self.current_char
        self.advance()

    if id_str == "div":
        return Token(TT_DIV_INT, pos_start=pos_start, pos_end=self.pos)
    
    if id_str == "mod":
        return Token(TT_MOD_INT, pos_start=pos_start, pos_end=self.pos)


    tok_type = TT_KEYWORD if id_str in KEYWORDS else TT_IDENTIFIER

    return Token(tok_type, id_str, pos_start, self.pos)
  
  def make_minus_or_arrow(self):
    tok_type = TT_MINUS
    pos_start = self.pos.copy()
    self.advance()

    if self.current_char == '>':
      self.advance()
      tok_type = TT_ARROW

    return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

  def make_not_equals(self):
    pos_start = self.pos.copy()
    self.advance()

    if self.current_char == '=':
      self.advance()
      return Token(TT_NE, pos_start=pos_start, pos_end=self.pos), None

    self.advance()
    return None, ExpectedCharError(pos_start, self.pos, "'=' (after '!')")
  
  def make_equals(self):
    tok_type = TT_EQ
    pos_start = self.pos.copy()
    self.advance()

    if self.current_char == '=':
      self.advance()
      tok_type = TT_EE

    return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

  def make_less_than(self):
    tok_type = TT_LT
    pos_start = self.pos.copy()
    self.advance()

    if self.current_char == '=':
      self.advance()
      tok_type = TT_LTE

    return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

  def make_greater_than(self):
    tok_type = TT_GT
    pos_start = self.pos.copy()
    self.advance()

    if self.current_char == '=':
      self.advance()
      tok_type = TT_GTE

    return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

  def skip_comment(self):
    self.advance()

    while self.current_char != '\n':
      self.advance()

class OutputNode:
    def __init__(self, expr_nodes, pos_start, pos_end):
        self.expr_nodes = expr_nodes
        self.pos_start = pos_start
        self.pos_end = pos_end

class InputNode:
    def __init__(self, expr_node, pos_start, pos_end):
        self.expr_node = expr_node
        self.pos_start = pos_start
        self.pos_end = pos_end

class NumberNode:
  def __init__(self, tok):
    self.tok = tok

    self.pos_start = self.tok.pos_start
    self.pos_end = self.tok.pos_end

  def __repr__(self):
    return f'{self.tok}'

class StringNode:
  def __init__(self, tok):
    self.tok = tok

    self.pos_start = self.tok.pos_start
    self.pos_end = self.tok.pos_end

  def __repr__(self):
    return f'{self.tok}'

class ListNode:
  def __init__(self, element_nodes, pos_start, pos_end):
    self.element_nodes = element_nodes

    self.pos_start = pos_start
    self.pos_end = pos_end

class ListAccessNode:
    def __init__(self, var_node, index_node):
        self.var_node = var_node
        self.index_node = index_node

        self.pos_start = var_node.pos_start
        self.pos_end = index_node.pos_end

class VarAccessNode:
  def __init__(self, var_name_tok):
    self.var_name_tok = var_name_tok

    self.pos_start = self.var_name_tok.pos_start
    self.pos_end = self.var_name_tok.pos_end

class VarAssignNode:
  def __init__(self, var_name_tok, value_node):
    self.var_name_tok = var_name_tok
    self.value_node = value_node

    self.pos_start = self.var_name_tok.pos_start
    self.pos_end = self.value_node.pos_end

class BinOpNode:
  def __init__(self, left_node, op_tok, right_node):
    self.left_node = left_node
    self.op_tok = op_tok
    self.right_node = right_node

    self.pos_start = self.left_node.pos_start
    self.pos_end = self.right_node.pos_end

  def __repr__(self):
    return f'({self.left_node}, {self.op_tok}, {self.right_node})'

class UnaryOpNode:
  def __init__(self, op_tok, node):
    self.op_tok = op_tok
    self.node = node

    self.pos_start = self.op_tok.pos_start
    self.pos_end = node.pos_end

  def __repr__(self):
    return f'({self.op_tok}, {self.node})'

class IfNode:
  def __init__(self, cases, else_case):
    self.cases = cases
    self.else_case = else_case

    self.pos_start = self.cases[0][0].pos_start
    self.pos_end = (self.else_case or self.cases[len(self.cases) - 1])[0].pos_end


class LoopNode:
    def __init__(self, var_name_tok, start_value_node, end_value_node, body_node, should_return_null):
    self.node_to_call = node_to_call
    self.arg_nodes = arg_nodes

    self.pos_start = self.node_to_call.pos_start

    if len(self.arg_nodes) > 0:
      self.pos_end = self.arg_nodes[len(self.arg_nodes) - 1].pos_end
    else:
      self.pos_end = self.node_to_call.pos_end

class ReturnNode:
  def __init__(self, node_to_return, pos_start, pos_end):
    self.node_to_return = node_to_return

    self.pos_start = pos_start
    self.pos_end = pos_end

class ContinueNode:
  def __init__(self, pos_start, pos_end):
    self.pos_start = pos_start
    self.pos_end = pos_end

class BreakNode:
  def __init__(self, pos_start, pos_end):
    self.pos_start = pos_start
    self.pos_end = pos_end

class MethodCallNode:
  def __init__(self, object_node, method_name, arg_nodes):
    self.object_node = object_node
    self.method_name = method_name
    self.arg_nodes = arg_nodes

    self.pos_start = self.object_node.pos_start
    if len(self.arg_nodes) > 0:
      self.pos_end = self.arg_nodes[len(self.arg_nodes) - 1].pos_end
    else:
      self.pos_end = self.method_name.pos_end

#######################################
# PARSE RESULT
#######################################

class ParseResult:
  def __init__(self):
    self.error = None
    self.node = None
    self.last_registered_advance_count = 0
    self.advance_count = 0
    self.to_reverse_count = 0

  def register_advancement(self):
    self.last_registered_advance_count = 1
    self.advance_count += 1

  def register(self, res):
    self.last_registered_advance_count = res.advance_count
    self.advance_count += res.advance_count
    if res.error: self.error = res.error
    return res.node

  def try_register(self, res):
    if res.error:
      self.to_reverse_count = res.advance_count
      return None
    return self.register(res)

  def success(self, node):
    self.node = node
    return self

  def failure(self, error):
    if not self.error or self.last_registered_advance_count == 0:
      self.error = error
    return self

#######################################
# PARSER
#######################################

class Parser:
  def __init__(self, tokens):
    self.tokens = tokens
    self.tok_idx = -1
    self.advance()

  def advance(self):
    self.tok_idx += 1
    self.update_current_tok()
    return self.current_tok

  def reverse(self, amount=1):
    self.tok_idx -= amount
    self.update_current_tok()
    return self.current_tok

  def update_current_tok(self):
    if self.tok_idx >= 0 and self.tok_idx < len(self.tokens):
      self.current_tok = self.tokens[self.tok_idx]

  def parse(self):
    res = self.statements()
    if not res.error and self.current_tok.type != TT_EOF:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Token cannot appear after previous tokens"
      ))
    return res

  ###################################

  def statements(self):
    res = ParseResult()
    statements = []
    pos_start = self.current_tok.pos_start.copy()

    while self.current_tok.type == TT_NEWLINE:
      res.register_advancement()
      self.advance()

    statement = res.register(self.statement())
    if res.error: return res
    statements.append(statement)

    more_statements = True

    while True:
      newline_count = 0
      while self.current_tok.type == TT_NEWLINE:
        res.register_advancement()
        self.advance()
        newline_count += 1
      if newline_count == 0:
        more_statements = False

      if not more_statements: break
      statement = res.try_register(self.statement())
      if not statement:
        self.reverse(res.to_reverse_count)
        more_statements = False
        continue
      statements.append(statement)

    return res.success(ListNode(
      statements,
      pos_start,
      self.current_tok.pos_end.copy()
    ))

  def statement(self):
    res = ParseResult()
    pos_start = self.current_tok.pos_start.copy()

    if self.current_tok.matches(TT_KEYWORD, 'return'):
      res.register_advancement()
      self.advance()

      expr = res.try_register(self.expr())
      if not expr:
        self.reverse(res.to_reverse_count)
      return res.success(ReturnNode(expr, pos_start, self.current_tok.pos_start.copy()))

    if self.current_tok.matches(TT_KEYWORD, 'continue'):
      res.register_advancement()
      self.advance()
      return res.success(ContinueNode(pos_start, self.current_tok.pos_start.copy()))
      
    if self.current_tok.matches(TT_KEYWORD, 'break'):
      res.register_advancement()
      self.advance()
      return res.success(BreakNode(pos_start, self.current_tok.pos_start.copy()))

    if self.current_tok.matches(TT_KEYWORD, 'input'):
            res.register_advancement()
            self.advance()

            if self.current_tok.type != TT_IDENTIFIER:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected identifier after 'input'"
                ))

            var_name_tok = self.current_tok
            res.register_advancement()
            self.advance()
            return res.success(InputNode(var_name_tok, pos_start, self.current_tok.pos_start.copy()))


    expr = res.register(self.expr())
    if res.error:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected 'RETURN', 'CONTINUE', 'BREAK', 'VAR', 'IF', 'FOR', 'WHILE', 'METHOD', int, float, identifier, '+', '-', '(', '[' or 'NOT'"
      ))
    return res.success(expr)

  def expr(self):
    res = ParseResult()

    if self.current_tok.type == TT_IDENTIFIER and self.current_tok.value.isupper():
        var_name = self.current_tok
        res.register_advancement()
        self.advance()

        if self.current_tok.type == TT_EQ:
            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            return res.success(VarAssignNode(var_name, expr))
        else:
            self.reverse()
    
    if self.current_tok.matches(TT_KEYWORD, 'output'):
      res.register_advancement()
      self.advance()
      pos_start = self.current_tok.pos_start.copy()
      expr_nodes = [res.register(self.expr())]
      if res.error: return res

      while self.current_tok.type == TT_COMMA:
          res.register_advancement()
          self.advance()
          expr_nodes.append(res.register(self.expr()))
          if res.error: return res

      return res.success(OutputNode(expr_nodes, pos_start, self.current_tok.pos_start.copy()))
      
    if self.current_tok.matches(TT_KEYWORD, 'input'):
        res.register_advancement()
        self.advance()
        expr = res.register(self.expr())
        if res.error: return res
        return res.success(InputNode(expr, self.current_tok.pos_start, self.current_tok.pos_start.copy()))

    node = res.register(self.bin_op(self.comp_expr, ((TT_KEYWORD, 'and'), (TT_KEYWORD, 'or'))))

    if res.error:
        return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            "Expected variable assignment or expression"
        ))

    return res.success(node)

  def comp_expr(self):
    res = ParseResult()

    if self.current_tok.matches(TT_KEYWORD, 'not'):
      op_tok = self.current_tok
      res.register_advancement()
      self.advance()

      node = res.register(self.comp_expr())
      if res.error: return res
      return res.success(UnaryOpNode(op_tok, node))
    
    node = res.register(self.bin_op(self.arith_expr, (TT_EE, TT_NE, TT_LT, TT_GT, TT_LTE, TT_GTE)))
    
    if res.error:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected int, float, identifier, '+', '-', '(', '[', 'IF', 'FOR', 'WHILE', 'METHOD' or 'NOT'"
      ))

    return res.success(node)

  def arith_expr(self):
    return self.bin_op(self.term, (TT_PLUS, TT_MINUS))

  def term(self):
    return self.bin_op(self.factor, (TT_MUL, TT_DIV, TT_DIV_INT, TT_MOD, TT_MOD_INT))
  
  def factor(self):
    res = ParseResult()
    tok = self.current_tok

    if tok.type in (TT_PLUS, TT_MINUS):
      res.register_advancement()
      self.advance()
      factor = res.register(self.factor())
      if res.error: return res
      return res.success(UnaryOpNode(tok, factor))

    return self.power()

  def power(self):
    return self.bin_op(self.call, (TT_POW, ), self.factor)

  def call(self):
    res = ParseResult()
    atom = res.register(self.atom())
    if res.error: return res

    if self.current_tok.type == TT_LPAREN:
      res.register_advancement()
      self.advance()
      arg_nodes = []

      if self.current_tok.type == TT_RPAREN:
        res.register_advancement()
        self.advance()
      else:
        arg_nodes.append(res.register(self.expr()))
        if res.error:
          return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            "Expected ')', 'VAR', 'IF', 'FOR', 'WHILE', 'FUN', int, float, identifier, '+', '-', '(', '[' or 'NOT'"
          ))

        while self.current_tok.type == TT_COMMA:
          res.register_advancement()
          self.advance()

          arg_nodes.append(res.register(self.expr()))
          if res.error: return res

        if self.current_tok.type != TT_RPAREN:
          return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            f"Expected ',' or ')'"
          ))

        res.register_advancement()
        self.advance()
      return res.success(CallNode(atom, arg_nodes))

    elif self.current_tok.type == TT_DOT:
            res.register_advancement()
            self.advance()
            
            if self.current_tok.type != TT_IDENTIFIER:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected identifier after '.'"
                ))
                
            method_name = self.current_tok
            res.register_advancement()
            self.advance()
            
            # Allow method calls with or without arguments
            if self.current_tok.type == TT_LPAREN:
                res.register_advancement()
                self.advance()
                arg_nodes = []
                
                if self.current_tok.type == TT_RPAREN:
                    res.register_advancement()
                    self.advance()
                else:
                    arg_nodes.append(res.register(self.expr()))
                    if res.error:
                        return res.failure(InvalidSyntaxError(
                            self.current_tok.pos_start, self.current_tok.pos_end,
                            "Expected ')', expression"
                        ))
                        
                    while self.current_tok.type == TT_COMMA:
                        res.register_advancement()
                        self.advance()
                        
                        arg_nodes.append(res.register(self.expr()))
                        if res.error: return res
                        
                    if self.current_tok.type != TT_RPAREN:
                        return res.failure(InvalidSyntaxError(
                            self.current_tok.pos_start, self.current_tok.pos_end,
                            "Expected ',' or ')'"
                        ))
                        
                    res.register_advancement()
                    self.advance()
                    
                # Return method call node
                atom = MethodCallNode(atom, method_name, arg_nodes)
            else:
                # Support property access (e.g., NUMBERS.length), but in your language, always require ()
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected '(' after method name"
                ))
    return res.success(atom)

  def atom(self):
    res = ParseResult()
    tok = self.current_tok

    if tok.type in (TT_INT, TT_FLOAT):
        res.register_advancement()
        self.advance()
        return res.success(NumberNode(tok))

    elif tok.type == TT_STRING:
        res.register_advancement()
        self.advance()
        return res.success(StringNode(tok))
    
    elif tok.matches(TT_KEYWORD, 'new'):
        res.register_advancement()
        self.advance()
        if self.current_tok.type == TT_IDENTIFIER:
            class_name_tok = self.current_tok
            res.register_advancement()
            self.advance()
            if self.current_tok.type == TT_LPAREN:
                res.register_advancement()
                self.advance()
                arg_nodes = []
                if self.current_tok.type == TT_RPAREN:
                    res.register_advancement()
                    self.advance()
                else:
                    arg_nodes.append(res.register(self.expr()))
                    if res.error: return res
                    while self.current_tok.type == TT_COMMA:
                        res.register_advancement()
                        self.advance()
                        arg_nodes.append(res.register(self.expr()))
                        if res.error: return res
                    if self.current_tok.type != TT_RPAREN:
                        return res.failure(InvalidSyntaxError(
                            self.current_tok.pos_start, self.current_tok.pos_end,
                            "Expected ',' or ')'"
                        ))
                    res.register_advancement()
                    self.advance()

                return res.success(CallNode(
                    VarAccessNode(Token(TT_IDENTIFIER, 'new', tok.pos_start, tok.pos_end)),
                    [VarAccessNode(class_name_tok)] + arg_nodes
                ))
            else:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected '(' after class name"
                ))
        else:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected class name after 'new'"
            ))

    elif tok.type == TT_IDENTIFIER:
        res.register_advancement()
        self.advance()
        node = VarAccessNode(tok)

        if self.current_tok.type == TT_LSQUARE:
            res.register_advancement()
            self.advance()
            index = res.register(self.expr())
            if res.error: return res

            if self.current_tok.type != TT_RSQUARE:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected ']'"
                ))

            res.register_advancement()
            self.advance()
            return res.success(ListAccessNode(node, index))

        return res.success(node)

    elif tok.type == TT_LPAREN:
        res.register_advancement()
        self.advance()
        expr = res.register(self.expr())
        if res.error: return res
        if self.current_tok.type == TT_RPAREN:
            res.register_advancement()
            self.advance()
            return res.success(expr)
        else:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected ')'"
            ))

    elif tok.type == TT_LSQUARE:
        list_expr = res.register(self.list_expr())
        if res.error: return res
        return res.success(list_expr)

    elif tok.matches(TT_KEYWORD, 'if'):
        if_expr = res.register(self.if_expr())
        if res.error: return res
        return res.success(if_expr)


    elif tok.matches(TT_KEYWORD, 'while'):
        while_expr = res.register(self.while_expr())
        if res.error: return res
        return res.success(while_expr)
    
    elif tok.matches(TT_KEYWORD, 'loop'):
        loop_expr = res.register(self.loop_expr())
        if res.error: return res
        return res.success(loop_expr)

    elif tok.matches(TT_KEYWORD, 'method'):
        func_def = res.register(self.func_def())
        if res.error: return res
        return res.success(func_def)

    return res.failure(InvalidSyntaxError(
        tok.pos_start, tok.pos_end,
        "Expected int, float, identifier, '+', '-', '(', '[', IF', 'WHILE', 'FUN'"
    ))
  def list_expr(self):
    res = ParseResult()
    element_nodes = []
    pos_start = self.current_tok.pos_start.copy()

    if self.current_tok.type != TT_LSQUARE:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected '['"
      ))
    res.register_advancement()
    self.advance()

    if self.current_tok.type == TT_RSQUARE:
      res.register_advancement()
      self.advance()
    else:
      element_nodes.append(res.register(self.expr()))
      if res.error:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          "Expected ']', 'VAR', 'IF', 'WHILE', 'FUN', int, float, identifier, '+', '-', '(', '[' or 'NOT'"
        ))

      while self.current_tok.type == TT_COMMA:
        res.register_advancement()
        self.advance()

        element_nodes.append(res.register(self.expr()))
        if res.error: return res

      if self.current_tok.type != TT_RSQUARE:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected ',' or ']'"
        ))

      res.register_advancement()
      self.advance()

    return res.success(ListNode(
      element_nodes,
      pos_start,
      self.current_tok.pos_end.copy()
    ))

  def if_expr(self):
    res = ParseResult()
    all_cases = res.register(self.if_expr_cases('if'))
    if res.error: return res
    cases, else_case = all_cases
    return res.success(IfNode(cases, else_case))

  def if_expr_b(self):
    return self.if_expr_cases('elif')
    
  def if_expr_c(self):
    res = ParseResult()
    else_case = None

    if self.current_tok.matches(TT_KEYWORD, 'else'):
      res.register_advancement()
      self.advance()

      if self.current_tok.type == TT_NEWLINE:
        res.register_advancement()
        self.advance()

        statements = res.register(self.statements())
        if res.error: return res
        else_case = (statements, True)

        if self.current_tok.matches(TT_KEYWORD, 'end'):
          res.register_advancement()
          self.advance()
        else:
          return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            "Expected 'END'"
          ))
        if self.current_tok.matches(TT_KEYWORD, 'if'):
          res.register_advancement()
          self.advance()
      else:
        expr = res.register(self.statement())
        if res.error: return res
        else_case = (expr, False)

    return res.success(else_case)

  def if_expr_b_or_c(self):
    res = ParseResult()
    cases, else_case = [], None

    if self.current_tok.matches(TT_KEYWORD, 'elif'):
      all_cases = res.register(self.if_expr_b())
      if res.error: return res
      cases, else_case = all_cases
    else:
      else_case = res.register(self.if_expr_c())
      if res.error: return res
    
    return res.success((cases, else_case))

  def if_expr_cases(self, case_keyword):
    res = ParseResult()
    cases = []
    else_case = None

    if not self.current_tok.matches(TT_KEYWORD, case_keyword):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected '{case_keyword}'"
      ))

    res.register_advancement()
    self.advance()

    condition = res.register(self.expr())
    if res.error: return res

    if not self.current_tok.matches(TT_KEYWORD, 'then'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'THEN'"
      ))

    res.register_advancement()
    self.advance()

    if self.current_tok.type == TT_NEWLINE:
      res.register_advancement()
      self.advance()

      statements = res.register(self.statements())
      if res.error: return res
      cases.append((condition, statements, True))

      if self.current_tok.matches(TT_KEYWORD, 'end'):
        res.register_advancement()
        self.advance()

        if self.current_tok.matches(TT_KEYWORD, 'if'):
          res.register_advancement()
          self.advance()
      else:
        all_cases = res.register(self.if_expr_b_or_c())
        if res.error: return res
        new_cases, else_case = all_cases
        cases.extend(new_cases)
    else:
      expr = res.register(self.statement())
      if res.error: return res
      cases.append((condition, expr, False))

      all_cases = res.register(self.if_expr_b_or_c())
      if res.error: return res
      new_cases, else_case = all_cases
      cases.extend(new_cases)

    return res.success((cases, else_case))

  
  def loop_expr(self):
    res = ParseResult()

    if not self.current_tok.matches(TT_KEYWORD, 'loop'):
        return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            f"Expected 'LOOP'"
        ))

    res.register_advancement()
    self.advance()

    if self.current_tok.matches(TT_KEYWORD, 'while'):
        res.register_advancement()
        self.advance()

        # Parse the condition after 'while'
        condition = res.register(self.expr())
        if res.error: return res

        # Expect a newline after the condition
        if self.current_tok.type != TT_NEWLINE:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected newline after loop condition"
            ))

        res.register_advancement()
        self.advance()

        # Parse the body of the loop
        body = res.register(self.statements())
        if res.error: return res

        # Expect 'end loop' to close the loop
        if not self.current_tok.matches(TT_KEYWORD, 'end'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected 'END'"
            ))

        res.register_advancement()
        self.advance()

        if not self.current_tok.matches(TT_KEYWORD, 'loop'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected 'LOOP'"
            ))

        res.register_advancement()
        self.advance()

        return res.success(WhileNode(condition, body, True))

    if self.current_tok.type == TT_IDENTIFIER:
      var_name = self.current_tok
      res.register_advancement()
      self.advance()

      if not self.current_tok.matches(TT_KEYWORD, 'from'):
          return res.failure(InvalidSyntaxError(
              self.current_tok.pos_start, self.current_tok.pos_end,
              f"Expected 'FROM'"
          ))

      res.register_advancement()
      self.advance()

      start_value = res.register(self.expr())
      if res.error: return res

      if not self.current_tok.matches(TT_KEYWORD, 'to'):
          return res.failure(InvalidSyntaxError(
              self.current_tok.pos_start, self.current_tok.pos_end,
              f"Expected 'TO'"
          ))

      res.register_advancement()
      self.advance()

      end_value = res.register(self.expr())
      if res.error: return res

      if self.current_tok.matches(TT_KEYWORD, 'then'):
          res.register_advancement()
          self.advance()
          

      if self.current_tok.type == TT_NEWLINE:
          res.register_advancement()
          self.advance()

          body = res.register(self.statements())
          if res.error: return res

          if not self.current_tok.matches(TT_KEYWORD, 'end'):
              return res.failure(InvalidSyntaxError(
                  self.current_tok.pos_start, self.current_tok.pos_end,
                  f"Expected 'END'"
              ))

          res.register_advancement()
          self.advance()

          if not self.current_tok.matches(TT_KEYWORD, 'loop'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected 'LOOP'"
            ))
          
          res.register_advancement()
          self.advance()
          return res.success(LoopNode(var_name, start_value, end_value, body, True))

      return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected 'THEN' or NEWLINE"
      ))


  def func_def(self):
    res = ParseResult()

    if not self.current_tok.matches(TT_KEYWORD, 'method'):
        return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            f"Expected 'fun'"
        ))

    res.register_advancement()
    self.advance()

    if self.current_tok.type == TT_IDENTIFIER:
        var_name_tok = self.current_tok
        res.register_advancement()
        self.advance()
        if self.current_tok.type != TT_LPAREN:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '('"
            ))
    else:
        var_name_tok = None
        if self.current_tok.type != TT_LPAREN:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected identifier or '('"
            ))

    res.register_advancement()
    self.advance()
    arg_name_toks = []

    while self.current_tok.type == TT_IDENTIFIER:
        arg_name_toks.append(self.current_tok)
        res.register_advancement()
        self.advance()

        if self.current_tok.type == TT_COMMA:
            res.register_advancement()
            self.advance()
        else:
            break

    if self.current_tok.type != TT_RPAREN:
        return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            f"Expected ',' or ')'"
        ))

    res.register_advancement()
    self.advance()

    if self.current_tok.type == TT_ARROW:
        res.register_advancement()
        self.advance()

        body = res.register(self.expr())
        if res.error: return res

        return res.success(FuncDefNode(
            var_name_tok,
            arg_name_toks,
            body,
            True
        ))

    if self.current_tok.type != TT_NEWLINE:
        return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            f"Expected '->' or NEWLINE"
        ))

    res.register_advancement()
    self.advance()

    body = res.register(self.statements())
    if res.error: return res

    if not self.current_tok.matches(TT_KEYWORD, 'end'):
        return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            f"Expected 'END'"
        ))

    res.register_advancement()
    self.advance()

    if not self.current_tok.matches(TT_KEYWORD, 'method'):
      return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            f"Expected 'method' after 'end'"
        ))
    
    res.register_advancement()
    self.advance()

    return res.success(FuncDefNode(
        var_name_tok,
        arg_name_toks,
        body,
        False
    ))
  ###################################

  def bin_op(self, func_a, ops, func_b=None):
    if func_b == None:
      func_b = func_a
    
    res = ParseResult()
    left = res.register(func_a())
    if res.error: return res

    while self.current_tok.type in ops or (self.current_tok.type, self.current_tok.value) in ops:
      op_tok = self.current_tok
      res.register_advancement()
      self.advance()
      right = res.register(func_b())
      if res.error: return res
      left = BinOpNode(left, op_tok, right)

    return res.success(left)

#######################################
# RUNTIME RESULT
#######################################

class RTResult:
  def __init__(self):
    self.reset()

  def reset(self):
    self.value = None
    self.error = None
    self.func_return_value = None
    self.loop_should_continue = False
    self.loop_should_break = False

  def register(self, res):
    self.error = res.error
    self.func_return_value = res.func_return_value
    self.loop_should_continue = res.loop_should_continue
    self.loop_should_break = res.loop_should_break
    return res.value

  def success(self, value):
    self.reset()
    self.value = value
    return self

  def success_return(self, value):
    self.reset()
    self.func_return_value = value
    return self
  
  def success_continue(self):
    self.reset()
    self.loop_should_continue = True
    return self

  def success_break(self):
    self.reset()
    self.loop_should_break = True
    return self

  def failure(self, error):
    self.reset()
    self.error = error
    return self

  def should_return(self):
    # Note: this will allow you to continue and break outside the current function
    return (
      self.error or
      self.func_return_value or
      self.loop_should_continue or
      self.loop_should_break
    )

#######################################
# VALUES
#######################################

class Value:
  def __init__(self):
    self.set_pos()
    self.set_context()

  def set_pos(self, pos_start=None, pos_end=None):
    self.pos_start = pos_start
    self.pos_end = pos_end
    return self

  def set_context(self, context=None):
    self.context = context
    return self

  def call_method(self, method_name, args):
    return None, self.illegal_operation()

  def added_to(self, other):
    return None, self.illegal_operation(other)

  def subbed_by(self, other):
    return None, self.illegal_operation(other)

  def multed_by(self, other):
    return None, self.illegal_operation(other)

  def dived_by(self, other):
    return None, self.illegal_operation(other)

  def powed_by(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_eq(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_ne(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_lt(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_gt(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_lte(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_gte(self, other):
    return None, self.illegal_operation(other)

  def anded_by(self, other):
    return None, self.illegal_operation(other)

  def ored_by(self, other):
    return None, self.illegal_operation(other)

  def notted(self, other):
    return None, self.illegal_operation(other)

  def execute(self, args):
    return RTResult().failure(self.illegal_operation())

  def copy(self):
    raise Exception('No copy method defined')

  def is_true(self):
    return False

  def illegal_operation(self, other=None):
    if not other: other = self
    return RTError(
      self.pos_start, other.pos_end,
      'Illegal operation',
      self.context
    )

class Number(Value):
  def __init__(self, value):
    super().__init__()
    self.value = value

  def call_method(self, method_name, args):
    if method_name == "to_string":
      return String(str(self.value)).set_context(self.context), None
    
    return None, RTError(
      None, None,
      f"Number has no method '{method_name}'",
      self.context
    )

  def added_to(self, other):
    if isinstance(other, Number):
      return Number(self.value + other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def subbed_by(self, other):
    if isinstance(other, Number):
      return Number(self.value - other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def multed_by(self, other):
    if isinstance(other, Number):
      return Number(self.value * other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def dived_by(self, other):
    if isinstance(other, Number):
      if other.value == 0:
        return None, RTError(
          other.pos_start, other.pos_end,
          'Division by zero',
          self.context
        )

      return Number(self.value / other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def powed_by(self, other):
    if isinstance(other, Number):
      return Number(self.value ** other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def modded_by(self, other):
    if isinstance(other, Number):
      return Number(self.value % other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparison_eq(self, other):
    if isinstance(other, Number):
      return Number(int(self.value == other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparison_ne(self, other):
    if isinstance(other, Number):
      return Number(int(self.value != other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparison_lt(self, other):
    if isinstance(other, Number):
      return Number(int(self.value < other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparison_gt(self, other):
    if isinstance(other, Number):
      return Number(int(self.value > other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparison_lte(self, other):
    if isinstance(other, Number):
      return Number(int(self.value <= other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparison_gte(self, other):
    if isinstance(other, Number):
      return Number(int(self.value >= other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def anded_by(self, other):
    if isinstance(other, Number):
      return Number(int(self.value and other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def ored_by(self, other):
    if isinstance(other, Number):
      return Number(int(self.value or other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def notted(self):
    return Number(1 if self.value == 0 else 0).set_context(self.context), None
  
  def dived_by_int(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(
                    self.pos_start, other.pos_end,
                    'Division by zero',
                    self.context
                )
            return Number(self.value // other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

  def modded_by_int(self, other):
        if isinstance(other, Number):
            return Number(self.value % other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)
      


  def copy(self):
    copy = Number(self.value)
    copy.set_pos(self.pos_start, self.pos_end)
    copy.set_context(self.context)
    return copy

  def is_true(self):
    return self.value != 0

  def __str__(self):
    return str(self.value)
  
  def __repr__(self):
    return str(self.value)

Number.null = Number(0)
Number.false = Number(0)
Number.true = Number(1)
Number.math_PI = Number(math.pi)

class String(Value):
  def __init__(self, value):
    super().__init__()
    self.value = value

  def call_method(self, method_name, args):
    if method_name == "length":
      return Number(len(self.value)).set_context(self.context), None
    elif method_name == "upper":
      return String(self.value.upper()).set_context(self.context), None
    elif method_name == "lower":
      return String(self.value.lower()).set_context(self.context), None
    elif method_name == "split":
      if len(args) > 0 and isinstance(args[0], String):
        parts = self.value.split(args[0].value)
        return List([String(part).set_context(self.context) for part in parts]).set_context(self.context), None
      else:
        parts = self.value.split()
        return List([String(part).set_context(self.context) for part in parts]).set_context(self.context), None
    
    return None, RTError(
      None, None,
      f"String has no method '{method_name}'",
      self.context
    )

  def added_to(self, other):
    if isinstance(other, String):
      return String(self.value + other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def multed_by(self, other):
    if isinstance(other, Number):
      return String(self.value * other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def is_true(self):
    return len(self.value) > 0

  def copy(self):
    copy = String(self.value)
    copy.set_pos(self.pos_start, self.pos_end)
    copy.set_context(self.context)
    return copy

  def __str__(self):
    return self.value

  def __repr__(self):
    return f'"{self.value}"'

def to_value(item):
        if isinstance(item, Value):
            return item
        elif isinstance(item, str):
            return String(item)
        elif isinstance(item, (int, float)):
            return Number(item)
        elif isinstance(item, list):
            return List([to_value(x) for x in item])
        return item

class Collection(Value):
    def __init__(self, *elements):
        super().__init__()
        self.items = [to_value(x) for x in elements]
        self._iter_index = 0
   
    def call_method(self, method_name, args):
        if method_name in ("add", "addItem"):
            if len(args) < 1:
                return None, RTError(
                    self.pos_start, self.pos_end,
                    f"'{method_name}' requires 1 argument", self.context
                )
            item = to_value(args[0])
            self.items.append(item)
            return Number(len(self.items)), None
        
        elif method_name == "hasNext":
            return Number(1 if self._iter_index < len(self.items) else 0), None
        
        elif method_name == "getNext":
            if self._iter_index < len(self.items):
                item = self.items[self._iter_index]
                self._iter_index += 1
                return item, None
            else:
                return Number.null, None
            
        elif method_name == "resetNext":
            self._iter_index = 0
            return Number.null, None
        
        elif method_name == "length":
            return Number(len(self.items)).set_context(self.context), None
        return None, RTError(None, None, f"Collection has no method '{method_name}'", self.context)

    def copy(self):
        c = Collection()
        c.items = [item.copy() for item in self.items]
        c._iter_index = self._iter_index
        c.set_context(self.context)
        c.set_pos(self.pos_start, self.pos_end)
        return c

    def __repr__(self):
        return f"<Collection {self.items}>"


class List(Value):
  def __init__(self, elements):
    super().__init__()
    self.elements = elements

  def call_method(self, method_name, args):
    if method_name == "length":
      return Number(len(self.elements)).set_context(self.context), None
    elif method_name == "append":
      if len(args) < 1:
        return None, RTError(
          None, None,
          "append() takes at least 1 argument",
          self.context
        )
      self.elements.append(args[0])
      return Number.null, None
    elif method_name == "pop":
      if len(self.elements) == 0:
        return None, RTError(
          None, None,
          "Cannot pop from empty list",
          self.context
        )
      if len(args) > 0 and isinstance(args[0], Number):
        try:
          index = int(args[0].value)
          element = self.elements.pop(index)
          return element, None
        except:
          return None, RTError(
            None, None,
            "Invalid index for pop()",
            self.context
          )
      else:
        return self.elements.pop(), None
    
    return None, RTError(
      None, None,
      f"List has no method '{method_name}'",
      self.context
    )

  def added_to(self, other):
    new_list = self.copy()
    new_list.elements.append(other)
    return new_list, None

  def subbed_by(self, other):
    if isinstance(other, Number):
      new_list = self.copy()
      try:
        new_list.elements.pop(other.value)
        return new_list, None
      except:
        return None, RTError(
          other.pos_start, other.pos_end,
          'Element at this index could not be removed from list because index is out of bounds',
          self.context
        )
    else:
      return None, Value.illegal_operation(self, other)

  def multed_by(self, other):
    if isinstance(other, List):
      new_list = self.copy()
      new_list.elements.extend(other.elements)
      return new_list, None
    else:
      return None, Value.illegal_operation(self, other)

  def dived_by(self, other):
    if isinstance(other, Number):
      try:
        return self.elements[other.value], None
      except:
        return None, RTError(
          other.pos_start, other.pos_end,
          'Element at this index could not be retrieved from list because index is out of bounds',
          self.context
        )
    else:
      return None, Value.illegal_operation(self, other)
  
  def copy(self):
    copy = List(self.elements)
    copy.set_pos(self.pos_start, self.pos_end)
    copy.set_context(self.context)
    return copy

  def __str__(self):
    return ", ".join([str(x) for x in self.elements])

  def __repr__(self):
    return f'[{", ".join([repr(x) for x in self.elements])}]'

class BaseFunction(Value):
  def __init__(self, name):
    super().__init__()
    self.name = name or "<anonymous>"

  def generate_new_context(self):
    new_context = Context(self.name, self.context, self.pos_start)
    new_context.symbol_table = SymbolTable(new_context.parent.symbol_table)
    return new_context

  def check_args(self, arg_names, args):
    res = RTResult()

    if len(args) > len(arg_names):
      return res.failure(RTError(
        self.pos_start, self.pos_end,
        f"{len(args) - len(arg_names)} too many args passed into {self}",
        self.context
      ))
    
    if len(args) < len(arg_names):
      return res.failure(RTError(
        self.pos_start, self.pos_end,
        f"{len(arg_names) - len(args)} too few args passed into {self}",
        self.context
      ))

    return res.success(None)

  def populate_args(self, arg_names, args, exec_ctx):
    for i in range(len(args)):
      arg_name = arg_names[i]
      arg_value = args[i]
      arg_value.set_context(exec_ctx)
      exec_ctx.symbol_table.set(arg_name, arg_value)

  def check_and_populate_args(self, arg_names, args, exec_ctx):
    res = RTResult()
    res.register(self.check_args(arg_names, args))
    if res.should_return(): return res
    self.populate_args(arg_names, args, exec_ctx)
    return res.success(None)

class Function(BaseFunction):
  def __init__(self, name, body_node, arg_names, should_auto_return):
    super().__init__(name)
    self.body_node = body_node
    self.arg_names = arg_names
    self.should_auto_return = should_auto_return

  def execute(self, args):
    res = RTResult()
    interpreter = Interpreter()
    exec_ctx = self.generate_new_context()

    res.register(self.check_and_populate_args(self.arg_names, args, exec_ctx))
    if res.should_return(): return res

    value = res.register(interpreter.visit(self.body_node, exec_ctx))
    if res.should_return() and res.func_return_value == None: return res

    ret_value = (value if self.should_auto_return else None) or res.func_return_value or Number.null
    return res.success(ret_value)

  def copy(self):
    copy = Function(self.name, self.body_node, self.arg_names, self.should_auto_return)
    copy.set_context(self.context)
    copy.set_pos(self.pos_start, self.pos_end)
    return copy

  def __repr__(self):
    return f"<function {self.name}>"

class BuiltInFunction(BaseFunction):
  def __init__(self, name):
    super().__init__(name)

  def execute(self, args):
    res = RTResult()
    exec_ctx = self.generate_new_context()

    method_name = f'execute_{self.name}'
    method = getattr(self, method_name, self.no_visit_method)

    res.register(self.check_and_populate_args(method.arg_names, args, exec_ctx))
    if res.should_return(): return res

    return_value = res.register(method(exec_ctx))
    if res.should_return(): return res
    return res.success(return_value)
  
  def no_visit_method(self, node, context):
    raise Exception(f'No execute_{self.name} method defined')

  def copy(self):
    copy = BuiltInFunction(self.name)
    copy.set_context(self.context)
    copy.set_pos(self.pos_start, self.pos_end)
    return copy

  def __repr__(self):
    return f"<built-in function {self.name}>"

  #####################################

  def execute_print(self, exec_ctx):
    print(str(exec_ctx.symbol_table.get('value')))
    return RTResult().success(Number.null)
  execute_print.arg_names = ['value']
  
  def execute_print_ret(self, exec_ctx):
    return RTResult().success(String(str(exec_ctx.symbol_table.get('value'))))
  execute_print_ret.arg_names = ['value']
  
  def execute_input(self, exec_ctx):
    text = input(exec_ctx.symbol_table.get('value'))
    if text.isnumeric(): return RTResult().success(Number(int(text)))
    return RTResult().success(String(text))
  execute_input.arg_names = ['value']

  def execute_clear(self, exec_ctx):
    os.system('cls' if os.name == 'nt' else 'cls') 
    return RTResult().success(Number.null)
  execute_clear.arg_names = []

  def execute_is_number(self, exec_ctx):
    is_number = isinstance(exec_ctx.symbol_table.get("value"), Number)
    return RTResult().success(Number.true if is_number else Number.false)
  execute_is_number.arg_names = ["value"]

  def execute_is_string(self, exec_ctx):
    is_number = isinstance(exec_ctx.symbol_table.get("value"), String)
    return RTResult().success(Number.true if is_number else Number.false)
  execute_is_string.arg_names = ["value"]

  def execute_is_list(self, exec_ctx):
    is_number = isinstance(exec_ctx.symbol_table.get("value"), List)
    return RTResult().success(Number.true if is_number else Number.false)
  execute_is_list.arg_names = ["value"]

  def execute_is_function(self, exec_ctx):
    is_number = isinstance(exec_ctx.symbol_table.get("value"), BaseFunction)
    return RTResult().success(Number.true if is_number else Number.false)
  execute_is_function.arg_names = ["value"]

  def execute_append(self, exec_ctx):
    list_ = exec_ctx.symbol_table.get("list")
    value = exec_ctx.symbol_table.get("value")

    if not isinstance(list_, List):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "First argument must be list",
        exec_ctx
      ))

    list_.elements.append(value)
    return RTResult().success(Number.null)
  execute_append.arg_names = ["list", "value"]

  def execute_pop(self, exec_ctx):
    list_ = exec_ctx.symbol_table.get("list")
    index = exec_ctx.symbol_table.get("index")

    if not isinstance(list_, List):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "First argument must be list",
        exec_ctx
      ))

    if not isinstance(index, Number):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "Second argument must be number",
        exec_ctx
      ))

    try:
      element = list_.elements.pop(index.value)
    except:
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        'Element at this index could not be removed from list because index is out of bounds',
        exec_ctx
      ))
    return RTResult().success(element)
  execute_pop.arg_names = ["list", "index"]

  def execute_extend(self, exec_ctx):
    listA = exec_ctx.symbol_table.get("listA")
    listB = exec_ctx.symbol_table.get("listB")

    if not isinstance(listA, List):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "First argument must be list",
        exec_ctx
      ))

    if not isinstance(listB, List):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "Second argument must be list",
        exec_ctx
      ))

    listA.elements.extend(listB.elements)
    return RTResult().success(Number.null)
  execute_extend.arg_names = ["listA", "listB"]

  def execute_len(self, exec_ctx):
    list_ = exec_ctx.symbol_table.get("list")

    if not isinstance(list_, List):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "Argument must be list",
        exec_ctx
      ))

    return RTResult().success(Number(len(list_.elements)))
  execute_len.arg_names = ["list"]

  def execute_run(self, exec_ctx):
    fn = exec_ctx.symbol_table.get("fn")

    if not isinstance(fn, String):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "Second argument must be string",
        exec_ctx
      ))

    fn = fn.value
    abs_path = os.path.abspath(fn)  # Ensure relative paths are resolved

    try:
      with open(abs_path, "r") as f:
        script = f.read()
    except Exception as e:
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        f"Failed to load script \"{fn}\"\n" + str(e),
        exec_ctx
      ))

    _, error = run(abs_path, script)
    
    if error:
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        f"Failed to finish executing script \"{fn}\"\n" +
        error.as_string(),
        exec_ctx
      ))

    return RTResult().success(Number.null)
  execute_run.arg_names = ["fn"]

  def execute_str(self, exec_ctx):
    return RTResult().success(String(str(exec_ctx.symbol_table.get("value"))))
  execute_str.arg_names = ["value"]

  def execute_int(self, exec_ctx):
    return RTResult().success(Number(int(str(exec_ctx.symbol_table.get("value")))))
  execute_int.arg_names = ["value"]

  def execute_random(self, exec_ctx):
        min_ = exec_ctx.symbol_table.get("min")
        max_ = exec_ctx.symbol_table.get("max")

        if not isinstance(min_, Number) or not isinstance(max_, Number):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Both arguments to random() must be numbers",
                exec_ctx
            ))

        value = pyrandom.randint(int(min_.value), int(max_.value))
        return RTResult().success(Number(value))
  execute_random.arg_names = ["min", "max"]


BuiltInFunction.print       = BuiltInFunction("print")
BuiltInFunction.print_ret   = BuiltInFunction("print_ret")
BuiltInFunction.input       = BuiltInFunction("input")
BuiltInFunction.clear       = BuiltInFunction("clear")
BuiltInFunction.is_number   = BuiltInFunction("is_number")
BuiltInFunction.is_string   = BuiltInFunction("is_string")
BuiltInFunction.is_list     = BuiltInFunction("is_list")
BuiltInFunction.is_function = BuiltInFunction("is_function")
BuiltInFunction.append      = BuiltInFunction("append")
BuiltInFunction.pop         = BuiltInFunction("pop")
BuiltInFunction.extend      = BuiltInFunction("extend")
BuiltInFunction.len					= BuiltInFunction("len")
BuiltInFunction.run					= BuiltInFunction("run")
BuiltInFunction.str					= BuiltInFunction("str")
BuiltInFunction.int         = BuiltInFunction("int")
BuiltInFunction.random      = BuiltInFunction("random")


#######################################
# CONTEXT
#######################################

class Context:
  def __init__(self, display_name, parent=None, parent_entry_pos=None):
    self.display_name = display_name
    self.parent = parent
    self.parent_entry_pos = parent_entry_pos
    self.symbol_table = None

#######################################
# SYMBOL TABLE
#######################################

class SymbolTable:
  def __init__(self, parent=None):
    self.symbols = {}
    self.parent = parent

  def get(self, name):
    value = self.symbols.get(name, None)
    if value == None and self.parent:
      return self.parent.get(name)
    return value

  def set(self, name, value):
    self.symbols[name] = value

  def remove(self, name):
    del self.symbols[name]

#######################################
# INTERPRETER
#######################################

class Interpreter:
  def visit(self, node, context):
      if isinstance(node, OutputNode): return self.visit_OutputNode(node, context)

      method_name = f'visit_{type(node).__name__}'
      method = getattr(self, method_name, self.no_visit_method)
      return method(node, context)

  def no_visit_method(self, node, context):
      raise Exception(f'No visit_{type(node).__name__} method defined')

  def visit_MethodCallNode(self, node, context):
    res = RTResult()
    
    # Get the object
    obj = res.register(self.visit(node.object_node, context))
    if res.should_return(): return res
    
    # Prepare method arguments
    args = []
    for arg_node in node.arg_nodes:
      args.append(res.register(self.visit(arg_node, context)))
      if res.should_return(): return res
    
    # Call the method on the object
    method_name = node.method_name.value
    result, error = obj.call_method(method_name, args)
    
    if error:
      return res.failure(error)
    
    return res.success(result.set_pos(node.pos_start, node.pos_end).set_context(context))

  def visit_LoopNode(self, node, context):
    res = RTResult()
    elements = []

    start_value = res.register(self.visit(node.start_value_node, context))
    if res.should_return(): return res

    end_value = res.register(self.visit(node.end_value_node, context))
    if res.should_return(): return res

    i = start_value.value

    while i < end_value.value:
        context.symbol_table.set(node.var_name_tok.value, Number(i))
        i += 1

        value = res.register(self.visit(node.body_node, context))
        if res.should_return() and res.loop_should_continue == False and res.loop_should_break == False: return res

        if res.loop_should_continue:
            continue

        if res.loop_should_break:
            break

        elements.append(value)

    return res.success(
        Number.null if node.should_return_null else
        List(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
    )

  ###################################
  
  def visit_OutputNode(self, node, context):
    res = RTResult()
    values = []

    for expr_node in node.expr_nodes:
        value = res.register(self.visit(expr_node, context))
        if res.should_return(): return res
        values.append(value)

    # Join values as strings with a space separator and print them
    print(" ".join(map(str, values)))
    return res.success(Number.null)
  
  def visit_InputNode(self, node, context):
    res = RTResult()
    uinput = input()
    value = String(uinput).set_context(context)
    if res.should_return(): return res 
    context.symbol_table.set(node.expr_node.value, value)

    return res.success(value)

  def visit_NumberNode(self, node, context):
    return RTResult().success(
      Number(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
    )

  def visit_StringNode(self, node, context):
    return RTResult().success(
      String(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
    )

  def visit_ListNode(self, node, context):
    res = RTResult()
    elements = []

    for element_node in node.element_nodes:
      elements.append(res.register(self.visit(element_node, context)))
      if res.should_return(): return res

    return res.success(
      List(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
    )

  def visit_ListAccessNode(self, node, context):
    res = RTResult()
    list_value = res.register(self.visit(node.var_node, context))
    if res.should_return(): return res

    index = res.register(self.visit(node.index_node, context))
    if res.should_return(): return res

    if not isinstance(list_value, List):
        return res.failure(RTError(
            node.pos_start, node.pos_end,
            "Expected list",
            context
        ))

    if not isinstance(index, Number):
        return res.failure(RTError(
            node.pos_start, node.pos_end,
            "List index must be a number",
            context
        ))

    try:
        return res.success(list_value.elements[index.value].copy().set_pos(node.pos_start, node.pos_end).set_context(context))
    except IndexError:
        return res.failure(RTError(
            node.pos_start, node.pos_end,
            "List index out of range",
            context
        ))
  def visit_LoopNode(self, node, context):
    res = RTResult()
    elements = []

    start_value = res.register(self.visit(node.start_value_node, context))
    if res.should_return(): return res

    end_value = res.register(self.visit(node.end_value_node, context))
    if res.should_return(): return res

    i = start_value.value

    while i < end_value.value:
        context.symbol_table.set(node.var_name_tok.value, Number(i))
        i += 1

        value = res.register(self.visit(node.body_node, context))
        if res.should_return() and res.loop_should_continue == False and res.loop_should_break == False: return res

        if res.loop_should_continue:
            continue

        if res.loop_should_break:
            break

        elements.append(value)

    return res.success(
        Number.null if node.should_return_null else
        List(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
    )
    
  def visit_VarAccessNode(self, node, context):
    res = RTResult()
    var_name = node.var_name_tok.value
    value = context.symbol_table.get(var_name)

    if not value:
      return res.failure(RTError(
        node.pos_start, node.pos_end,
        f"'{var_name}' is not defined",
        context
      ))

    if isinstance(value, (Number, String)):
        value = value.copy().set_pos(node.pos_start, node.pos_end).set_context(context)
    else:
        value = value.set_pos(node.pos_start, node.pos_end).set_context(context)
    return res.success(value)

  def visit_VarAssignNode(self, node, context):
    res = RTResult()
    var_name = node.var_name_tok.value
    value = res.register(self.visit(node.value_node, context))
    if res.should_return(): return res

    context.symbol_table.set(var_name, value)
    return res.success(value)

  def visit_BinOpNode(self, node, context):
    res = RTResult()
    left = res.register(self.visit(node.left_node, context))
    if res.should_return(): return res
    right = res.register(self.visit(node.right_node, context))
    if res.should_return(): return res

    if node.op_tok.type == TT_PLUS:
      result, error = left.added_to(right)
    elif node.op_tok.type == TT_MINUS:
      result, error = left.subbed_by(right)
    elif node.op_tok.type == TT_MUL:
      result, error = left.multed_by(right)
    elif node.op_tok.type == TT_DIV:
      result, error = left.dived_by(right)
    elif node.op_tok.type == TT_POW:
      result, error = left.powed_by(right)
    elif node.op_tok.type == TT_EE:
      result, error = left.get_comparison_eq(right)
    elif node.op_tok.type == TT_NE:
      result, error = left.get_comparison_ne(right)
    elif node.op_tok.type == TT_LT:
      result, error = left.get_comparison_lt(right)
    elif node.op_tok.type == TT_GT:
      result, error = left.get_comparison_gt(right)
    elif node.op_tok.type == TT_LTE:
      result, error = left.get_comparison_lte(right)
    elif node.op_tok.type == TT_GTE:
      result, error = left.get_comparison_gte(right)
    elif node.op_tok.type == TT_MOD:
      result, error = left.modded_by(right)
    elif node.op_tok.type == TT_MOD_INT:
      result, error = left.modded_by_int(right)
      if error: return res.failure(error)
      return res.success(result)
    elif node.op_tok.type == TT_DIV_INT:
      result, error = left.dived_by_int(right)
      if error: return res.failure(error)
      return res.success(result)
    elif node.op_tok.matches(TT_KEYWORD, 'and'):
      result, error = left.anded_by(right)
    elif node.op_tok.matches(TT_KEYWORD, 'or'):
      result, error = left.ored_by(right)
    

    if error:
      return res.failure(error)
    else:
      return res.success(result.set_pos(node.pos_start, node.pos_end))

  def visit_UnaryOpNode(self, node, context):
    res = RTResult()
    number = res.register(self.visit(node.node, context))
    if res.should_return(): return res

    error = None

    if node.op_tok.type == TT_MINUS:
      number, error = number.multed_by(Number(-1))
    elif node.op_tok.matches(TT_KEYWORD, 'not'):
      number, error = number.notted()

    if error:
      return res.failure(error)
    else:
      return res.success(number.set_pos(node.pos_start, node.pos_end))

  def visit_IfNode(self, node, context):
    res = RTResult()

    for condition, expr, should_return_null in node.cases:
      condition_value = res.register(self.visit(condition, context))
      if res.should_return(): return res

      if condition_value.is_true():
        expr_value = res.register(self.visit(expr, context))
        if res.should_return(): return res
        return res.success(Number.null if should_return_null else expr_value)

    if node.else_case:
      expr, should_return_null = node.else_case
      expr_value = res.register(self.visit(expr, context))
      if res.should_return(): return res
      return res.success(Number.null if should_return_null else expr_value)

    return res.success(Number.null)

  def visit_ForNode(self, node, context):
    res = RTResult()
    elements = []

    start_value = res.register(self.visit(node.start_value_node, context))
    if res.should_return(): return res

    end_value = res.register(self.visit(node.end_value_node, context))
    if res.should_return(): return res

    if node.step_value_node:
      step_value = res.register(self.visit(node.step_value_node, context))
      if res.should_return(): return res
    else:
      step_value = Number(1)

    i = start_value.value

    if step_value.value >= 0:
      condition = lambda: i < end_value.value
    else:
      condition = lambda: i > end_value.value
    
    while condition():
      context.symbol_table.set(node.var_name_tok.value, Number(i))
      i += step_value.value

      value = res.register(self.visit(node.body_node, context))
      if res.should_return() and res.loop_should_continue == False and res.loop_should_break == False: return res
      
      if res.loop_should_continue:
        continue
      
      if res.loop_should_break:
        break

      elements.append(value)

    return res.success(
      Number.null if node.should_return_null else
      List(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
    )

  def visit_WhileNode(self, node, context):
    res = RTResult()
    elements = []

    while True:
      condition = res.register(self.visit(node.condition_node, context))
      if res.should_return(): return res

      if not condition.is_true():
        break

      value = res.register(self.visit(node.body_node, context))
      if res.should_return() and res.loop_should_continue == False and res.loop_should_break == False: return res

      if res.loop_should_continue:
        continue
      
      if res.loop_should_break:
        break

      elements.append(value)

    return res.success(
      Number.null if node.should_return_null else
      List(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
    )

  def visit_FuncDefNode(self, node, context):
    res = RTResult()

    func_name = node.var_name_tok.value if node.var_name_tok else None
    body_node = node.body_node
    arg_names = [arg_name.value for arg_name in node.arg_name_toks]
    func_value = Function(func_name, body_node, arg_names, node.should_auto_return).set_context(context).set_pos(node.pos_start, node.pos_end)
    
    if node.var_name_tok:
      context.symbol_table.set(func_name, func_value)

    return res.success(func_value)

  def visit_CallNode(self, node, context):
    res = RTResult()
    if (
        isinstance(node.node_to_call, VarAccessNode)
        and node.node_to_call.var_name_tok.value == "new"
        and len(node.arg_nodes) >= 1
        and isinstance(node.arg_nodes[0], VarAccessNode)
        and node.arg_nodes[0].var_name_tok.value == "Collection"
    ):
        # Evaluate all arguments after 'Collection'
        elements = [res.register(self.visit(arg_node, context)) for arg_node in node.arg_nodes[1:]]
        if res.should_return(): return res
        return res.success(Collection(*elements).set_context(context).set_pos(node.pos_start, node.pos_end))
    args = []

    value_to_call = res.register(self.visit(node.node_to_call, context))
    if res.should_return(): return res
    value_to_call = value_to_call.copy().set_pos(node.pos_start, node.pos_end)

    for arg_node in node.arg_nodes:
      args.append(res.register(self.visit(arg_node, context)))
      if res.should_return(): return res

    return_value = res.register(value_to_call.execute(args))
    if res.should_return(): return res
    return_value = return_value.copy().set_pos(node.pos_start, node.pos_end).set_context(context)
    return res.success(return_value)

  def visit_ReturnNode(self, node, context):
    res = RTResult()

    if node.node_to_return:
      value = res.register(self.visit(node.node_to_return, context))
      if res.should_return(): return res
    else:
      value = Number.null
    
    return res.success_return(value)


  def visit_ContinueNode(self, node, context):
    return RTResult().success_continue()

  def visit_BreakNode(self, node, context):
    return RTResult().success_break()

#######################################
# RUN
#######################################

global_symbol_table = SymbolTable()
global_symbol_table.set("null", Number.null)
global_symbol_table.set("false", Number.false)
global_symbol_table.set("true", Number.true)
global_symbol_table.set("math_PI", Number.math_PI)
global_symbol_table.set("print", BuiltInFunction.print)
global_symbol_table.set("print_ret", BuiltInFunction.print_ret)
global_symbol_table.set("input", BuiltInFunction.input)
global_symbol_table.set("clear", BuiltInFunction.clear)
global_symbol_table.set("cls", BuiltInFunction.clear)
global_symbol_table.set("is_number", BuiltInFunction.is_number)
global_symbol_table.set("is_string", BuiltInFunction.is_string)
global_symbol_table.set("is_list", BuiltInFunction.is_list)
global_symbol_table.set("is_function", BuiltInFunction.is_function)
global_symbol_table.set("append", BuiltInFunction.append)
global_symbol_table.set("pop", BuiltInFunction.pop)
global_symbol_table.set("extend", BuiltInFunction.extend)
global_symbol_table.set("len", BuiltInFunction.len)
global_symbol_table.set("run", BuiltInFunction.run)
global_symbol_table.set("str", BuiltInFunction.str)
global_symbol_table.set("int", BuiltInFunction.int)
global_symbol_table.set("random", BuiltInFunction.random)


def run(fn, text):
  """
  Run BASIC code with comprehensive error reporting
  
  Args:
      fn: Filename (for error reporting)
      text: Source code to execute
      
  Returns:
      tuple: (result, error)
  """
  try:
    # Generate tokens
    lexer = Lexer(fn, text)
    tokens, error = lexer.make_tokens()
    if error: 
      print(f"Lexer Error - Failed to tokenize input in {fn}")
      return None, error
    
    # Generate AST
    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: 
      print(f"Parser Error - Failed to parse tokens in {fn}")
      return None, ast.error

    # Run program
    interpreter = Interpreter()
    context = Context('<program>')
    context.symbol_table = global_symbol_table
    result = interpreter.visit(ast.node, context)

    return result.value, result.error
    
  except Exception as e:
    # Catch any uncaught Python exceptions and convert to language errors
    import traceback
    print(f"Internal Error - Unhandled exception in the interpreter")
    print(traceback.format_exc())
    # Create a dummy position for internal errors
    pos = Position(0, 0, 0, fn, text)
    return None, RTError(
      pos, pos,
      f"Internal interpreter error: {type(e).__name__}: {str(e)}",
      Context('<internal>')
    )

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python language.py <filename>")
        sys.exit(1)
    filename = sys.argv[1]
    try:
        with open(filename, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as e:
        print(f"Could not open file '{filename}': {e}")
        sys.exit(1)
    result, error = run(filename, code)
    if error:
        print(error.as_string())
    # Output is handled by the language's output statement