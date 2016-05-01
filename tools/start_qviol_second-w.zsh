
data=example/bell/
prefix=example/qviol-bell3x2/

ts -S 4

# Primer:
# ts qviol $data/3x2-fin-08D.txt -d 222 -o $data/qviol-bell3x2-08D-222-none.yml
# ts qviol $data/3x2-fin-08D.txt -d 333 -o $data/qviol-bell3x2-08D-333-none.yml
# ts qviol $data/3x2-fin-12D.txt -d 222 -o $data/qviol-bell3x2-12D-222-none.yml
# ts qviol $data/3x2-fin-12D.txt -d 333 -o $data/qviol-bell3x2-12D-333-none.yml
# ts qviol $data/3x2-fin-18D.txt -d 222 -o $data/qviol-bell3x2-18D-222-none.yml
# ts qviol $data/3x2-fin-18D.txt -d 333 -o $data/qviol-bell3x2-18D-333-none.yml
# ts qviol $data/3x2-fin-26D.txt -d 222 -o $data/qviol-bell3x2-26D-222-none.yml
# ts qviol $data/3x2-fin-26D.txt -d 333 -o $data/qviol-bell3x2-26D-333-none.yml

# 08D
ts qviol ${prefix}08D-222-none.yml -d 222 -p w -o ${prefix}08D-222-none-w.yml
ts qviol ${prefix}08D-222-none.yml -d 222 -p w -o ${prefix}08D-222-CHSHE-w.yml -c CHSHE
ts qviol ${prefix}08D-222-none.yml -d 222 -p w -o ${prefix}08D-222-CHSH-w.yml -c CHSH
ts qviol ${prefix}08D-222-none.yml -d 222 -p w -o ${prefix}08D-222-PPT-w.yml -c PPT

# 12D
ts qviol ${prefix}12D-222-none.yml -d 222 -p w -o ${prefix}12D-222-none-w.yml
ts qviol ${prefix}12D-222-none.yml -d 222 -p w -o ${prefix}12D-222-CHSHE-w.yml -c CHSHE
ts qviol ${prefix}12D-222-none.yml -d 222 -p w -o ${prefix}12D-222-CHSH-w.yml -c CHSH
ts qviol ${prefix}12D-222-none.yml -d 222 -p w -o ${prefix}12D-222-PPT-w.yml -c PPT

# 14D
ts qviol ${prefix}14D-222-none.yml -d 222 -p w -o ${prefix}14D-222-none-w.yml
ts qviol ${prefix}14D-222-none.yml -d 222 -p w -o ${prefix}14D-222-CHSHE-w.yml -c CHSHE
ts qviol ${prefix}14D-222-none.yml -d 222 -p w -o ${prefix}14D-222-CHSH-w.yml -c CHSH
ts qviol ${prefix}14D-222-none.yml -d 222 -p w -o ${prefix}14D-222-PPT-w.yml -c PPT

# 18D
ts qviol ${prefix}18D-222-none.yml -d 222 -p w -o ${prefix}18D-222-none-w.yml
ts qviol ${prefix}18D-222-none.yml -d 222 -p w -o ${prefix}18D-222-CHSHE-w.yml -c CHSHE
ts qviol ${prefix}18D-222-none.yml -d 222 -p w -o ${prefix}18D-222-CHSH-w.yml -c CHSH
ts qviol ${prefix}18D-222-none.yml -d 222 -p w -o ${prefix}18D-222-PPT-w.yml -c PPT

# 26D
ts qviol ${prefix}26D-222-none.yml -d 222 -p w -o ${prefix}26D-222-none-w.yml
ts qviol ${prefix}26D-222-none.yml -d 222 -p w -o ${prefix}26D-222-CHSHE-w.yml -c CHSHE
ts qviol ${prefix}26D-222-none.yml -d 222 -p w -o ${prefix}26D-222-CHSH-w.yml -c CHSH
ts qviol ${prefix}26D-222-none.yml -d 222 -p w -o ${prefix}26D-222-PPT-w.yml -c PPT
