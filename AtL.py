import string
from format_errors import *
# CONSTANTS
DIGITS = '0123456789'
ALPHAs = string.ascii_letters;
# TOKENS
AtL_INT = 'INT'
AtL_IDENTIFIER = 'IDENTIFIER'
AtL_KEYWORD = 'KEYWORD'
AtL_FLOAT = 'FLOAT'
AtL_PLUS = 'PLUS'
AtL_MIN = 'MIN'
AtL_MUL = 'MUL'
AtL_DIV = 'DIV'
AtL_SUB = 'SUB'
AtL_LPAREN = 'LPAREN'
AtL_RPAREN = 'RPAREN'
AtL_EQ = 'EQ'
AtL_EEQ = 'EEQ'
AtL_NQ = 'NQ'
AtL_LT = 'LT'
AtL_GT = 'GT'
AtL_LTE = 'LTE'
AtL_GTE = 'GTE'
AtL_LcPAREN = 'LcPAREN'
AtL_RcPAREN = 'RcPAREN'
AtL_EOF = 'EOF'

KEYWORDS = [
    'var',
    'and',
    'or',
    'not',
    'if',
    'elseif',
    'else',
    'while'
    'then'
]

class ErrorClass:
    def __init__(self,pos_st,pos_end,type,details=None):
        self.pos_start = pos_st
        self.pos_end = pos_end
        self.type = type
        self.details = details
    def as_string(self):
        error = f'{self.type}:{self.details}'
        error += f' File: {self.pos_start.fname}, ' \
                 f'line: {self.pos_start.line+1}, ' \
                 f'col: {self.pos_start.col+1}'
        error += '\n\n'+highlight_error(self.pos_start.ftext,self.pos_start,self.pos_end)
        return  error

class IllegaleChar(ErrorClass):
    def __init__(self,pos_st,pos_end,details):
        super().__init__(pos_st,pos_end,'Illegal Character',details)
class InvalidSyntaxErr(ErrorClass):
    def __init__(self,pos_st,pos_end,details):
        super().__init__(pos_st,pos_end,'Invalid Syntax', details)

class RunTimeErr(ErrorClass):
    def __init__(self,pos_st,pos_end,details):
        super().__init__(pos_st,pos_end,'Run time error:', details)


class Position:
    def __init__(self,idx,line,col,fname,ftext):
        self.idx =idx
        self.line = line
        self.col = col
        self.fname = fname
        self.ftext = ftext
    def advance(self,curr_chr=None):
        self.idx +=1
        self.col +=1

        if curr_chr == "\n":
            self.line +=1
            self.col = 0
        return self
    def copy(self):
        return Position(self.idx,self.line,self.col,self.fname,self.ftext)

class ParseResults:
    def __init__(self):
        self.error = None
        self.node = None
    def register(self,res):
        if isinstance(res,ParseResults):
            if res.error: self.error = res.error
            return res.node
        return res
    def success(self,node):
        self.node = node
        return self
    def failure(self,err):
        self.error = err
        return self

class Token:
    def __init__(self,type,value=None,start_pos=None,end_pos=None):
        self.type = type
        self.value = value
        if start_pos:
            self.pos_start = start_pos.copy()
            self.pos_end = start_pos.copy()
            self.pos_end.advance()
        if end_pos:
            self.pos_end = end_pos

    def __repr__(self):
        if self.value: return f'{self.type}:{self.value}'
        else:
            return f'{self.type}'

    def matches(self,type_,value):
        return self.type==type_ and self.value==value
# Lexical Analyser (Take input text --> return its tokens)
class LexicalAnalyser:
    def __init__(self,fname,text):
        self.text = text
        self.pos = Position(-1,0,-1,fname,text)
        self.curr_char = None
        self.advance()
    def advance(self):
        self.pos.advance(self.curr_char)
        self.curr_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

    def make_tokens(self):
        tokens = []
        while self.curr_char != None:
            if self.curr_char in ' \t':
                self.advance()
            elif self.curr_char in DIGITS:
                tokens.append(self.make_number())
            elif self.curr_char in ALPHAs:
                tokens.append(self.make_str())
            elif self.curr_char =='+':
                tokens.append(Token(AtL_PLUS,start_pos=self.pos))
                self.advance()
            elif self.curr_char =='-':
                tokens.append(Token(AtL_MIN,start_pos=self.pos))
                self.advance()
            elif self.curr_char =='*':
                tokens.append(Token(AtL_MUL,start_pos=self.pos))
                self.advance()
            elif self.curr_char =='(':
                tokens.append(Token(AtL_LPAREN,start_pos=self.pos))
                self.advance()
            elif self.curr_char ==')':
                tokens.append(Token(AtL_RPAREN,start_pos=self.pos))
                self.advance()
            elif self.curr_char =='{':
                tokens.append(Token(AtL_LcPAREN,start_pos=self.pos))
                self.advance()
            elif self.curr_char =='}':
                tokens.append(Token(AtL_RcPAREN,start_pos=self.pos))
                self.advance()
            elif self.curr_char =='/':
                tokens.append(Token(AtL_DIV,start_pos=self.pos))
                self.advance()
            elif self.curr_char =='!':
                tok, err = self.make_not_equal()
                if err: return [], err
                tokens.append(tok)
            elif self.curr_char == '=':
                tokens.append(self.make_equal())
            elif self.curr_char == '<':
                tokens.append(self.make_lessThan())
            elif self.curr_char == '>':
                tokens.append(self.make_greaterThan())

            else:
                pos_st = self.pos.copy()
                char =self.curr_char
                return [],IllegaleChar(pos_st,self.pos,"'"+char+"'")
        tokens.append(Token(AtL_EOF,start_pos=self.pos))
        return tokens, None
    def make_number(self):
        num_str = ''
        dot_count =0
        pos_start = self.pos.copy()
        while self.curr_char != None and self.curr_char in DIGITS+'.':
            if self.curr_char == '.':
                if dot_count ==1: break
                dot_count+=1
                num_str+='.'
            else:
                num_str+= self.curr_char
            self.advance()
        if dot_count==0:
            return Token(AtL_INT,int(num_str),pos_start,self.pos)
        else:
            return Token(AtL_FLOAT, float(num_str),pos_start,self.pos)

    def make_str(self):
        ch_str = ''
        pos_st = self.pos.copy()
        while self.curr_char != None and self.curr_char in ALPHAs+DIGITS+'_':
            ch_str+= self.curr_char
            self.advance()
        if ch_str in KEYWORDS:
            tok_type = AtL_KEYWORD
        else:
            tok_type = AtL_IDENTIFIER

        return Token(tok_type, ch_str, pos_st, self.pos)

    def make_not_equal(self):
        pos_start = self.pos.copy()
        self.advance()
        if self.curr_char =='=':
            self.advance()
            return Token(AtL_NQ,pos_start,end_pos=self.pos),None
        self.advance()
        return None, InvalidSyntaxErr(pos_start,self.pos,"Expected '=' after '!'")
    def make_equal(self):
        tok_type = AtL_EQ
        pos_start = self.pos.copy()
        self.advance()
        if self.curr_char=='=':
            self.advance()
            tok_type = AtL_EEQ
        return Token(tok_type,pos_start,self.pos)
    def make_lessThan(self):
        tok_type = AtL_LT
        pos_start = self.pos.copy()
        self.advance()
        if self.curr_char=='=':
            self.advance()
            tok_type = AtL_LTE
        return Token(tok_type,pos_start,self.pos)
    def make_greaterThan(self):
        tok_type = AtL_GT
        pos_start = self.pos.copy()
        self.advance()
        if self.curr_char=='=':
            self.advance()
            tok_type = AtL_GTE
        return Token(tok_type,pos_start,self.pos)

class NumberNode:
    def __init__(self,tok):
        self.tok = tok
        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end
    def __repr__(self):
        return f'{self.tok}'

class VarAccessNode:
    def __init__(self,var_name_tok):
        self.var_name_tok = var_name_tok
        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.var_name_tok.pos_end
class VarAssignNode:
    def __init__(self,var_name_tok,value_node):
        self.var_name_tok = var_name_tok
        self.value_node = value_node
        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.var_name_tok.pos_end

class BinOpNode:
    def __init__(self,left_node,op_tok,right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node
        self.pos_start = left_node.pos_start
        self.pos_end = right_node.pos_end
    def __repr__(self):
        return f'({self.left_node},{self.op_tok},{self.right_node})'
class UnaryOpNode:
    def __init__(self,op_tok,node):
        self.op_tok = op_tok
        self.node = node
        self.pos_start = self.op_tok.pos_start
        self.pos_end = node.pos_end
    def __repr__(self):
        return f'({self.op_tok},{self.node})'
class IfNode:
    def __init__(self,cases,else_case):
        self.cases = cases
        self.else_case = else_case
        self.pos_start = self.cases[0][0].pos_start
        self.pos_end = self.else_case or self.cases[len(cases)-1][0].pos_end
class WhileNode:
	def __init__(self, condition_node, body_node):
		self.condition_node = condition_node
		self.body_node = body_node
		self.pos_start = self.condition_node.pos_start
		self.pos_end = self.body_node.pos_end
class Parser:
    def __init__(self,tokens):
        self.tokens = tokens
        self.tok_idx = -1
        self.advance()
    def advance(self):
        self.tok_idx +=1
        if self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]
        return self.current_tok

    def parse(self):
        res = self.expr()
        if not res.error and self.current_tok.type != AtL_EOF:
            return res.failure(InvalidSyntaxErr(
                self.current_tok.pos_start,self.current_tok.pos_end,"Expected '+', '-', '*' or '/'"
            ))
        return res

    def factor(self):
        res = ParseResults()
        tok = self.current_tok
        if tok.type in (AtL_PLUS,AtL_MIN):
            res.register(self.advance())
            factor = res.register(self.factor())
            if res.error:return res
            return res.success(UnaryOpNode(tok,factor))
        elif tok.type == AtL_IDENTIFIER:
            res.register(self.advance())
            return res.success(VarAccessNode(tok))
        elif tok.type in (AtL_INT, AtL_FLOAT):
            res.register(self.advance())
            return res.success(NumberNode(tok))
        elif tok.type ==AtL_LPAREN:
            res.register(self.advance())
            expr = res.register(self.expr())
            if res.error:return res
            if self.current_tok.type==AtL_RPAREN:
                res.register(self.advance())
                return res.success(expr)
            else:
                return res.failure(InvalidSyntaxErr(
                    self.current_tok.pos_start,self.current_tok.pos_end,"Expected ')'"
                ))
        # elif tok.matches(tt_keyword)

        return res.failure(InvalidSyntaxErr(
            tok.pos_start,tok.pos_end,"Expected int or float numbers"
        ))
    def term(self):
        return self.bin_op(self.factor,(AtL_DIV,AtL_MUL))
    def arith_expr(self):
        return self.bin_op(self.term,(AtL_PLUS,AtL_MIN))
    def comp_expr(self):
        res = ParseResults()
        if self.current_tok.matches(AtL_KEYWORD,'not'):
            op_tok = self.current_tok
            res.register(self.advance())
            node = res.register(self.comp_expr())
            if res.error: return res
            return res.success(UnaryOpNode(op_tok,node))
        node =res.register(self.bin_op(self.arith_expr,(AtL_EEQ,AtL_NQ,AtL_GT,AtL_GTE,AtL_GTE,AtL_LT,AtL_LTE)))
        if res.error:
            return res.failure(InvalidSyntaxErr(self.current_tok.pos_start,self.current_tok.pos_end,"Expected int or float numbers, '(', 'not'"))
        return res.success(node)
    def expr(self):
        res = ParseResults()
        if self.current_tok.matches(AtL_KEYWORD,'var') :
            res.register(self.advance())
            if self.current_tok.type != AtL_IDENTIFIER:
                return res.failure(InvalidSyntaxErr(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected Identifier"
                ))
            var_name = self.current_tok
            res.register(self.advance())
            if self.current_tok.type != AtL_EQ:
                return res.failure(InvalidSyntaxErr(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected '='"
                ))
            res.register(self.advance())
            expr = res.register(self.expr())
            if res.error: return res
            return res.success(VarAssignNode(var_name,expr))
        elif self.current_tok.type == AtL_IDENTIFIER :
            var_name = self.current_tok
            res.register(self.advance())
            if self.current_tok.type == AtL_EQ :
                res.register(self.advance())
                expr = res.register(self.expr())
                if res.error: return res
                return res.success(VarAssignNode(var_name,expr))
            return self.bin_op(self.comp_expr, ((AtL_KEYWORD, "and"), (AtL_KEYWORD, "or")))

        elif self.current_tok.matches(AtL_KEYWORD,'if'):
            if_expr = res.register(self.if_expr())
            if res.error: return res
            return res.success(if_expr)
        return self.bin_op(self.comp_expr, ((AtL_KEYWORD,"and"), (AtL_KEYWORD,"or")))
    def if_expr(self):
        res = ParseResults()
        cases =[]
        else_case = None
        if not self.current_tok.matches(AtL_KEYWORD, 'if'):
            return res.failure(InvalidSyntaxErr(
                self.current_tok.pos_start, self.current_tok.pos_end, "Expected 'if'"
            ))
        res.register(self.advance())
        if not self.current_tok.type == AtL_LPAREN:
            return res.failure(InvalidSyntaxErr(
                self.current_tok.pos_start, self.current_tok.pos_end, "Expected '('"
            ))
        res.register(self.advance())
        condition = res.register(self.expr())
        if res.error: return res

        #res.register(self.advance())
        if not self.current_tok.type == AtL_RPAREN:
            return res.failure(InvalidSyntaxErr(
                self.current_tok.pos_start, self.current_tok.pos_end, "Expected ')'"
            ))
        res.register(self.advance())
        if not self.current_tok.type == AtL_LcPAREN:
            return res.failure(InvalidSyntaxErr(
                    self.current_tok.pos_start, self.current_tok.pos_end, "Expected '{'"
            ))
        res.register(self.advance())
        statment = res.register(self.expr())
        if res.error: return res

        #res.register(self.advance())
        if not self.current_tok.type == AtL_RcPAREN:
            return res.failure(InvalidSyntaxErr(
                self.current_tok.pos_start, self.current_tok.pos_end, "Expected '}'"
            ))
        res.register(self.advance())
        cases.append((condition,statment))
        while self.current_tok.matches(AtL_KEYWORD,'elseif'):
            res.register(self.advance())
            if not self.current_tok.type == AtL_LPAREN:
                return res.failure(InvalidSyntaxErr(
                    self.current_tok.pos_start, self.current_tok.pos_end, "Expected '('"
                ))
            res.register(self.advance())
            condition = res.register(self.expr())
            if res.error: return res

            res.register(self.advance())
            if not self.current_tok.type == AtL_RPAREN:
                return res.failure(InvalidSyntaxErr(
                    self.current_tok.pos_start, self.current_tok.pos_end, "Expected ')'"
                ))
            res.register(self.advance())
            if not self.current_tok.type == AtL_LcPAREN:
                return res.failure(InvalidSyntaxErr(
                    self.current_tok.pos_start, self.current_tok.pos_end, "Expected '{'"
                ))
            res.register(self.advance())
            statment = res.register(self.expr())
            if res.error: return res

            res.register(self.advance())
            if not self.current_tok.type == AtL_RcPAREN:
                return res.failure(InvalidSyntaxErr(
                    self.current_tok.pos_start, self.current_tok.pos_end, "Expected '}'"
                ))
            cases.append((condition, statment))

        # return res.success(expr)
        if self.current_tok.matches(AtL_KEYWORD, 'else'):
            res.register(self.advance())
            if not self.current_tok.type == AtL_LcPAREN:
                return res.failure(InvalidSyntaxErr(
                    self.current_tok.pos_start, self.current_tok.pos_end, "Expected '{'"
                ))
            res.register(self.advance())
            expr = res.register(self.expr())
            if not self.current_tok.type == AtL_RcPAREN:
                return res.failure(InvalidSyntaxErr(
                    self.current_tok.pos_start, self.current_tok.pos_end, "Expected '}'"
                ))
            res.register(self.advance())
            if res.error: return res
            else_case = expr
        return res.success(IfNode(cases,else_case))

    def while_expr(self):
        res = ParseResults()

        if not self.current_tok.matches(AtL_KEYWORD, 'while'):
            return res.failure(InvalidSyntaxErr(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected 'WHILE'"
            ))

        res.register(self.advance())
        condition = res.register(self.expr())
        if res.error: return res

        if not self.current_tok.matches(AtL_KEYWORD, 'then'):
            return res.failure(InvalidSyntaxErr(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected 'THEN'"
            ))

        res.register(self.advance())
        body = res.register(self.expr())
        if res.error: return res

        return res.success(WhileNode(condition, body))

    def bin_op(self,func,ops):
        res = ParseResults()
        left = res.register(func())
        if res.error: return res
        while self.current_tok.type in ops or (self.current_tok.type,self.current_tok.value) in ops:
            op_tok = self.current_tok
            res.register(self.advance())
            right = res.register(func())
            if res.error:return res
            left = BinOpNode(left, op_tok, right)
        return res.success(left)

class Number:
    def __init__(self,value):
        self.value = value
        self.set_position()
    def set_position(self,pos_start=None, post_end=None):
        self.pos_start = pos_start
        self.pos_end = post_end
        return  self
    def added_to(self,other_num):
        if isinstance(other_num,Number):
            return Number(self.value+other_num.value),None
    def subtracted_from(self,other_num):
        if isinstance(other_num,Number):
            return Number(self.value-other_num.value),None
    def divided_by(self,other_num):
        if isinstance(other_num,Number):
            if other_num.value ==0:
                return None, RunTimeErr(other_num.pos_start,other_num.pos_end,'Division by zero')
            return Number(self.value/other_num.value),None
    def multiplied_to(self,other_num):
        if isinstance(other_num,Number):
            return Number(self.value*other_num.value),None
    def get_comparison_eq(self,other_num):
        if isinstance(other_num,Number):
            return Number(int(self.value==other_num.value)),None
    def get_comparison_eq(self,other_num):
        if isinstance(other_num,Number):
            return Number(int(self.value==other_num.value)),None
    def get_comparison_ne(self,other_num):
        if isinstance(other_num,Number):
            return Number(int(self.value!=other_num.value)),None
    def get_comparison_lt(self,other_num):
        if isinstance(other_num,Number):
            return Number(int(self.value<other_num.value)),None
    def get_comparison_lte(self,other_num):
        if isinstance(other_num,Number):
            return Number(int(self.value<=other_num.value)),None
    def get_comparison_gt(self,other_num):
        if isinstance(other_num,Number):
            return Number(int(self.value>other_num.value)),None
    def get_comparison_gte(self,other_num):
        if isinstance(other_num,Number):
            return Number(int(self.value>=other_num.value)),None
    def anded_by(self,other_num):
        if isinstance(other_num,Number):
            return Number(int(self.value and other_num.value)),None
    def ored_by(self,other_num):
        if isinstance(other_num,Number):
            return Number(int(self.value or other_num.value)),None
    def noted(self):
        return Number(1 if self.value == 0 else 0),None
    def is_true(self):
        return self.value !=0
    def __repr__(self):
        return str(self.value)


class RunResults:
    def __init__(self):
        self.error = None
        self.value = None
    def register(self,res):
        if res.error: self.error = res.error
        return res.value
    def success(self,value):
        self.value = value
        return self
    def failure(self,err):
        self.error = err
        return self
class Context:
    def __init__(self):
        self.symbol_table = None
class Interpreter:
    def execute(self,node,context):
        method_name = f'execute_{type(node).__name__}'
        method = getattr(self,method_name,self.no_execute_method)
        return method(node,context)
    def no_execute_method(self,node,context):
        raise Exception(f'No execute_{type(node).__name__} method defined')
    def execute_NumberNode(self,node,context):
        return RunResults().success(Number(node.tok.value).set_position(node.pos_start,node.pos_end))

    def execute_BinOpNode(self, node,context):
        res = RunResults()
        left = res.register(self.execute(node.left_node,context))
        if res.error:return res
        right = res.register(self.execute(node.right_node,context))
        if res.error: return res
        if node.op_tok.type ==AtL_PLUS:
            result,err = left.added_to(right)
        elif node.op_tok.type ==AtL_MIN:
            result,err = left.subtracted_from(right)
        elif node.op_tok.type ==AtL_MUL:
            result,err = left.multiplied_to(right)
        elif node.op_tok.type ==AtL_DIV:
            result,err = left.divided_by(right)
        elif node.op_tok.type ==AtL_EEQ:
            result,err = left.get_comparison_eq(right)
        elif node.op_tok.type ==AtL_GT:
            result,err = left.get_comparison_gt(right)
        elif node.op_tok.type ==AtL_GTE:
            result,err = left.get_comparison_gte(right)
        elif node.op_tok.type ==AtL_LT:
            result,err = left.get_comparison_lt(right)
        elif node.op_tok.type ==AtL_LTE:
            result,err = left.get_comparison_lte(right)
        elif node.op_tok.type ==AtL_GTE:
            result,err = left.get_comparison_gte(right)
        elif node.op_tok.type ==AtL_GTE:
            result,err = left.get_comparison_gte(right)
        elif node.op_tok.matches(AtL_KEYWORD,'and'):
            result,err = left.anded_by(right)
        elif node.op_tok.matches(AtL_KEYWORD,'or'):
            result,err = left.ored_by(right)

        if err:
            return res.failure(err)
        else:
            return res.success(result.set_position(node.pos_start,node.pos_end))

    def execute_UnaryOpNode(self, node,context):
        res = RunResults()
        number = res.register(self.execute(node.node,context))
        if res.error:return res
        err = None
        if node.op_tok.type == AtL_MIN:
            number,err = number.multiplied_to(Number(-1))
        elif node.op_tok.matches(AtL_KEYWORD,'not'):
            number, err = number.noted()

        if err:
            return res.failure(err)
        else:
            return res.success(number.set_position(node.pos_start,node.pos_end))

    def execute_VarAccessNode(self,node,context):
        res = RunResults()
        var_name = node.var_name_tok.value
        value = context.symbol_table.get(var_name)
        if not value:
            return res.failure(
                node.pos_start,node.pos_end,
                f"'{var_name}' is not defined"
            )
        return res.success(value)

    def execute_VarAssignNode(self,node,context):
        res = RunResults()
        var_name = node.var_name_tok.value
        value = res.register(self.execute(node.value_node,context))
        if res.error: return res

        context.symbol_table.set(var_name,value)
        return res.success(value)
    def execute_IfNode(self,node,context):
        res = RunResults()
        for condition,expr in node.cases:
            condition_value = res.register(self.execute(condition, context))
            if res.error: return res
            if condition_value.is_true():
                expr_value = res.register(self.execute(expr, context))
                if res.error: return res
                return res.success(expr_value)
        if node.else_case:
            else_value = res.register(self.execute(node.else_case, context))
            if res.error: return res
            return res.success(else_value)
        return res.success(None)
    def execute_WhileNode(self,node,context):
        res = RunResults()
        while True:
            condition = res.register(self.execute(node.condition_node, context))
            if res.error: return res
            if not condition.is_true(): break
            res.register(self.execute(node.body_node, context))
            if res.error: return res
        return res.success(None)

class SymbolTable:
    def __init__(self):
        self.symbols = {}
        self.parent = None
    def get(self,name):
        value = self.symbols.get(name,None)
        if value== None and self.parent:
            return self.parent.get(name)
        return value
    def set(self,name,value):
        self.symbols[name] = value
    def remove(self,name):
        del self.symbols[name]

global_symbol_table = SymbolTable()
global_symbol_table.set("null",Number(0))
global_symbol_table.set("true",Number(1))
global_symbol_table.set("false",Number(0))
def run(fn,text):
    lexer = LexicalAnalyser(fn,text)
    tokens,errors = lexer.make_tokens()
    if errors: return None,errors
    # AST
    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: return None, ast.error
    #Run Interpreter:
    interpreter = Interpreter()
    context = Context()
    context.symbol_table = global_symbol_table
    result = interpreter.execute(ast.node,context)

    return result.value,result.error