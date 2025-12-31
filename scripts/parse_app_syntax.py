import ast, sys
p='backend/app.py'
s=open(p, 'r', encoding='utf-8').read()
try:
    ast.parse(s)
    print('OK')
except SyntaxError as e:
    print('SyntaxError:', e)
    ln=e.lineno
    lines=s.splitlines()
    start=max(0, ln-5)
    end=min(len(lines), ln+5)
    for i in range(start, end):
        print(f"{i+1}: {lines[i]}")
    sys.exit(1)
