
data=example/bell/
prefix=example/qviol-bell3x2/
ts -S 4

ts qviol ${prefix}26D-333-none.yml -d 333 -o ${prefix}26D-333-PPT.yml -c PPT
ts qviol ${prefix}18D-333-none.yml -d 333 -o ${prefix}18D-333-PPT.yml -c PPT
ts qviol ${prefix}14D-333-none.yml -d 333 -o ${prefix}14D-333-PPT.yml -c PPT
ts qviol ${prefix}12D-333-none.yml -d 333 -o ${prefix}12D-333-PPT.yml -c PPT
ts qviol ${prefix}08D-333-none.yml -d 333 -o ${prefix}08D-333-PPT.yml -c PPT
