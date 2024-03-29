
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
ts qviol ${prefix}08D-222-none.yml -d 222 -p ghz -o ${prefix}08D-222-none-ghz.yml
ts qviol ${prefix}08D-222-none.yml -d 222 -p ghz -o ${prefix}08D-222-CHSHE-ghz.yml -c CHSHE
ts qviol ${prefix}08D-222-none.yml -d 222 -p ghz -o ${prefix}08D-222-CHSH-ghz.yml -c CHSH
ts qviol ${prefix}08D-222-none.yml -d 222 -p ghz -o ${prefix}08D-222-PPT-ghz.yml -c PPT

# 12D
ts qviol ${prefix}12D-222-none.yml -d 222 -p ghz -o ${prefix}12D-222-none-ghz.yml
ts qviol ${prefix}12D-222-none.yml -d 222 -p ghz -o ${prefix}12D-222-CHSHE-ghz.yml -c CHSHE
ts qviol ${prefix}12D-222-none.yml -d 222 -p ghz -o ${prefix}12D-222-CHSH-ghz.yml -c CHSH
ts qviol ${prefix}12D-222-none.yml -d 222 -p ghz -o ${prefix}12D-222-PPT-ghz.yml -c PPT

# 14D
ts qviol ${prefix}14D-222-none.yml -d 222 -p ghz -o ${prefix}14D-222-none-ghz.yml
ts qviol ${prefix}14D-222-none.yml -d 222 -p ghz -o ${prefix}14D-222-CHSHE-ghz.yml -c CHSHE
ts qviol ${prefix}14D-222-none.yml -d 222 -p ghz -o ${prefix}14D-222-CHSH-ghz.yml -c CHSH
ts qviol ${prefix}14D-222-none.yml -d 222 -p ghz -o ${prefix}14D-222-PPT-ghz.yml -c PPT

# 18D
ts qviol ${prefix}18D-222-none.yml -d 222 -p ghz -o ${prefix}18D-222-none-ghz.yml
ts qviol ${prefix}18D-222-none.yml -d 222 -p ghz -o ${prefix}18D-222-CHSHE-ghz.yml -c CHSHE
ts qviol ${prefix}18D-222-none.yml -d 222 -p ghz -o ${prefix}18D-222-CHSH-ghz.yml -c CHSH
ts qviol ${prefix}18D-222-none.yml -d 222 -p ghz -o ${prefix}18D-222-PPT-ghz.yml -c PPT

# 26D
ts qviol ${prefix}26D-222-none.yml -d 222 -p ghz -o ${prefix}26D-222-none-ghz.yml
ts qviol ${prefix}26D-222-none.yml -d 222 -p ghz -o ${prefix}26D-222-CHSHE-ghz.yml -c CHSHE
ts qviol ${prefix}26D-222-none.yml -d 222 -p ghz -o ${prefix}26D-222-CHSH-ghz.yml -c CHSH
ts qviol ${prefix}26D-222-none.yml -d 222 -p ghz -o ${prefix}26D-222-PPT-ghz.yml -c PPT

ts qviol ${prefix}26D-333-none.yml -d 333 -p ghz -o ${prefix}26D-333-none-ghz.yml
ts qviol ${prefix}26D-333-none.yml -d 333 -p ghz -o ${prefix}26D-333-CHSHE-ghz.yml -c CHSHE
ts qviol ${prefix}26D-333-none.yml -d 333 -p ghz -o ${prefix}26D-333-CGLMP-ghz.yml -c CGLMP
ts qviol ${prefix}26D-333-none.yml -d 333 -p ghz -o ${prefix}26D-333-PPT-ghz.yml -c PPT

ts qviol ${prefix}18D-333-none.yml -d 333 -p ghz -o ${prefix}18D-333-none-ghz.yml
ts qviol ${prefix}18D-333-none.yml -d 333 -p ghz -o ${prefix}18D-333-CHSHE-ghz.yml -c CHSHE
ts qviol ${prefix}18D-333-none.yml -d 333 -p ghz -o ${prefix}18D-333-CGLMP-ghz.yml -c CGLMP
ts qviol ${prefix}18D-333-none.yml -d 333 -p ghz -o ${prefix}18D-333-PPT-ghz.yml -c PPT

ts qviol ${prefix}14D-333-none.yml -d 333 -p ghz -o ${prefix}14D-333-none-ghz.yml
ts qviol ${prefix}14D-333-none.yml -d 333 -p ghz -o ${prefix}14D-333-CHSHE-ghz.yml -c CHSHE
ts qviol ${prefix}14D-333-none.yml -d 333 -p ghz -o ${prefix}14D-333-CGLMP-ghz.yml -c CGLMP
ts qviol ${prefix}14D-333-none.yml -d 333 -p ghz -o ${prefix}14D-333-PPT-ghz.yml -c PPT

ts qviol ${prefix}12D-333-none.yml -d 333 -p ghz -o ${prefix}12D-333-none-ghz.yml
ts qviol ${prefix}12D-333-none.yml -d 333 -p ghz -o ${prefix}12D-333-CHSHE-ghz.yml -c CHSHE
ts qviol ${prefix}12D-333-none.yml -d 333 -p ghz -o ${prefix}12D-333-CGLMP-ghz.yml -c CGLMP
ts qviol ${prefix}12D-333-none.yml -d 333 -p ghz -o ${prefix}12D-333-PPT-ghz.yml -c PPT

ts qviol ${prefix}08D-333-none.yml -d 333 -p ghz -o ${prefix}08D-333-none-ghz.yml
ts qviol ${prefix}08D-333-none.yml -d 333 -p ghz -o ${prefix}08D-333-CHSHE-ghz.yml -c CHSHE
ts qviol ${prefix}08D-333-none.yml -d 333 -p ghz -o ${prefix}08D-333-CGLMP-ghz.yml -c CGLMP
ts qviol ${prefix}08D-333-none.yml -d 333 -p ghz -o ${prefix}08D-333-PPT-ghz.yml -c PPT
