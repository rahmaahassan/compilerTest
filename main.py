import AtL

while True:
    text = input('Rahma Zakaria>')
    # print(text)
    result, error = AtL.run(text)
    if error: print(error.as_string())
    else:
        print(result)
