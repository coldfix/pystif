
prefix=example/qviol-bell3x2/

ts -S 4

# Primer:
# ts qviol $data/final-bell3x2-08D.txt -d 222 -o $data/qviol-bell3x2-08D-222-none.yml
# ts qviol $data/final-bell3x2-08D.txt -d 333 -o $data/qviol-bell3x2-08D-333-none.yml
# ts qviol $data/final-bell3x2-12D.txt -d 222 -o $data/qviol-bell3x2-12D-222-none.yml
# ts qviol $data/final-bell3x2-12D.txt -d 333 -o $data/qviol-bell3x2-12D-333-none.yml
# ts qviol $data/final-bell3x2-18D-rfd.txt -d 222 -o $data/qviol-bell3x2-18D-222-none.yml
# ts qviol $data/final-bell3x2-18D-rfd.txt -d 333 -o $data/qviol-bell3x2-18D-333-none.yml
# ts qviol $data/final-bell3x2-26D-p2f.txt -d 222 -o $data/qviol-bell3x2-26D-222-none.yml
# ts qviol $data/final-bell3x2-26D-p2f.txt -d 333 -o $data/qviol-bell3x2-26D-333-none.yml

# 08D
ts qviol ${prefix}08D-222-none.yml -d 222 -o ${prefix}08D-222-CHSHE.yml -c CHSHE
ts qviol ${prefix}08D-222-none.yml -d 222 -o ${prefix}08D-222-CHSH.yml -c CHSH
ts qviol ${prefix}08D-222-none.yml -d 222 -o ${prefix}08D-222-PPT.yml -c PPT

ts qviol ${prefix}08D-333-none.yml -d 333 -o ${prefix}08D-333-CHSHE.yml -c CHSHE
ts qviol ${prefix}08D-333-none.yml -d 333 -o ${prefix}08D-333-CGLMP.yml -c CGLMP

# 12D
ts qviol ${prefix}12D-222-none.yml -d 222 -o ${prefix}12D-222-CHSHE.yml -c CHSHE
ts qviol ${prefix}12D-222-none.yml -d 222 -o ${prefix}12D-222-CHSH.yml -c CHSH
ts qviol ${prefix}12D-222-none.yml -d 222 -o ${prefix}12D-222-PPT.yml -c PPT

ts qviol ${prefix}12D-333-none.yml -d 333 -o ${prefix}12D-333-CHSHE.yml -c CHSHE
ts qviol ${prefix}12D-333-none.yml -d 333 -o ${prefix}12D-333-CGLMP.yml -c CGLMP

# 14D
ts qviol ${prefix}14D-222-none.yml -d 222 -o ${prefix}14D-222-CHSHE.yml -c CHSHE
ts qviol ${prefix}14D-222-none.yml -d 222 -o ${prefix}14D-222-CHSH.yml -c CHSH
ts qviol ${prefix}14D-222-none.yml -d 222 -o ${prefix}14D-222-PPT.yml -c PPT

ts qviol ${prefix}14D-333-none.yml -d 333 -o ${prefix}14D-333-CHSHE.yml -c CHSHE
ts qviol ${prefix}14D-333-none.yml -d 333 -o ${prefix}14D-333-CGLMP.yml -c CGLMP

# 18D
ts qviol ${prefix}18D-222-none.yml -d 222 -o ${prefix}18D-222-CHSHE.yml -c CHSHE
ts qviol ${prefix}18D-222-none.yml -d 222 -o ${prefix}18D-222-CHSH.yml -c CHSH
ts qviol ${prefix}18D-222-none.yml -d 222 -o ${prefix}18D-222-PPT.yml -c PPT

ts qviol ${prefix}18D-333-none.yml -d 333 -o ${prefix}18D-333-CHSHE.yml -c CHSHE
ts qviol ${prefix}18D-333-none.yml -d 333 -o ${prefix}18D-333-CGLMP.yml -c CGLMP

# 26D
ts qviol ${prefix}26D-222-none.yml -d 222 -o ${prefix}26D-222-CHSHE.yml -c CHSHE
ts qviol ${prefix}26D-222-none.yml -d 222 -o ${prefix}26D-222-CHSH.yml -c CHSH
ts qviol ${prefix}26D-222-none.yml -d 222 -o ${prefix}26D-222-PPT.yml -c PPT

ts qviol ${prefix}26D-333-none.yml -d 333 -o ${prefix}26D-333-CHSHE.yml -c CHSHE
ts qviol ${prefix}26D-333-none.yml -d 333 -o ${prefix}26D-333-CGLMP.yml -c CGLMP