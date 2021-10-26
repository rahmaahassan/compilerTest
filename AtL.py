import string

# CONSTANTS
DIGITS = '0123456789'
ALPHAs = 'I, F' # Add ascii alphabets here using python string lib.
# TOKENS
AtL_INT = 'INT'
AtL_FLOAT = 'FLOAT'
AtL_PLUS = 'PLUS'
AtL_MIN = 'MIN'
AtL_MUL = 'MUL'
AtL_DIV = 'DIV'
AtL_SUB = 'SUB'
AtL_LPAREN = 'LPAREN'
AtL_RPAREN = 'RPAREN'
AtL_EQ = 'EQ'
class ErrorClass:
    def __init__(self,type,details=None):
        self.type = type
        self.details = details
    def as_string(self):
        error = f'{self.type}:{self.details}'
        return  error
class IllegaleChar(ErrorClass):
    def __init__(self,details):
        super().__init__('Illegal Character', details)


class Token:
    def __init__(self,type,value=None):
        self.type = type
        self.value = value
    def __repr__(self):
        if self.value: return f'{self.type}:{self.value}'
        else:
            return f'{self.type}'
# Lexical Analyser (Take input text --> return its tokens)
class LexicalAnalyser:
    def __init__(self,text):
        self.text = text
        self.pos = -1
        self.curr_char = None
        self.advance()
    def advance(self):
        self.pos +=1
        self.curr_char = self.text[self.pos] if self.pos < len(self.text) else None

    def make_tokens(self):
        tokens = []
        while self.curr_char != None:
            if self.curr_char in ' \t':
                self.advance()
            elif self.curr_char in DIGITS:
                tokens.append(self.make_number())
            elif self.curr_char in ALPHAs: # Here write your code
                tok,err = self.make_str(AtL_INT) # check for identifier (e.g. INT)
                if tok:
                    tokens.append(tok)
                else:
                    self.advance()
                    return tok,err
                # self.advance()
            elif self.curr_char =='+':
                tokens.append(Token(AtL_PLUS))
                self.advance()
            elif self.curr_char =='-':
                tokens.append(Token(AtL_MIN))
                self.advance()
            elif self.curr_char =='*':
                tokens.append(Token(AtL_MUL))
                self.advance()
            elif self.curr_char =='=':
                tokens.append(Token(AtL_EQ))
                self.advance()
            elif self.curr_char =='(':
                tokens.append(Token(AtL_LPAREN))
                self.advance()
            elif self.curr_char ==')':
                tokens.append(Token(AtL_RPAREN))
                self.advance()
            # Add the = sign
            else:
                char =self.curr_char
                self.advance()
                return [],IllegaleChar("'"+char+"'")
        return tokens, None
    def make_number(self):
        num_str = ''
        dot_count =0
        while self.curr_char != None and self.curr_char in DIGITS+'.':
            if self.curr_char == '.':
                if dot_count ==1: break
                dot_count+=1
                num_str+='.'
            else:
                num_str+= self.curr_char
            self.advance()
        if dot_count==0:
            return Token(AtL_INT,int(num_str))
        else:
            return Token(AtL_FLOAT, float(num_str))

    def make_str(self,operator):
        ch_str = ''
        count = 0
        while self.curr_char != None and self.curr_char in AtL_INT:
            if self.curr_char == '.':
                if count ==1: break
                count+=1
                ch_str+='.'
            else:
                ch_str+= self.curr_char
            self.advance()
        if ch_str==operator:
            return Token(ch_str, 'Identifier'),[] # Write your code
        else:
            return [],IllegaleChar("'"+ch_str+"'")
def run(text):
    lexer = LexicalAnalyser(text)
    tokens,errors = lexer.make_tokens()
    return tokens,errors