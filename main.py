import AtL

while True:
    text = input('@L>')
    # print(text)
    result,error = AtL.run('<stdin>',text)
    if error: print(error.as_string())
    elif result:
        print(result)