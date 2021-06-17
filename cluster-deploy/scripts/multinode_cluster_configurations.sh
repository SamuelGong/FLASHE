#!/bin/bash

user=ubuntu
deploy_dir=/data/projects/fate
db_auth=(root fate_dev)
redis_password=fate_dev
cxx_compile_flag=false

# services for 0
p0_mysql=172.32.40.185
p0_redis=172.32.40.185
p0_fate_flow=172.32.40.185
p0_fateboard=172.32.40.185
p0_federation=172.32.40.185
p0_proxy=172.32.40.185
p0_roll=172.32.40.185
p0_metaservice=172.32.40.185
p0_egg=172.32.40.185

# services for 1
p1_mysql=172.32.31.240
p1_redis=172.32.31.240
p1_fate_flow=172.32.31.240
p1_fateboard=172.32.31.240
p1_federation=172.32.31.240
p1_proxy=172.32.31.240
p1_roll=172.32.31.240
p1_metaservice=172.32.31.240
p1_egg=172.32.31.240

# services for 2
p2_mysql=172.32.11.116
p2_redis=172.32.11.116
p2_fate_flow=172.32.11.116
p2_fateboard=172.32.11.116
p2_federation=172.32.11.116
p2_proxy=172.32.11.116
p2_roll=172.32.11.116
p2_metaservice=172.32.11.116
p2_egg=172.32.11.116

# services for 3
p3_mysql=172.32.5.231
p3_redis=172.32.5.231
p3_fate_flow=172.32.5.231
p3_fateboard=172.32.5.231
p3_federation=172.32.5.231
p3_proxy=172.32.5.231
p3_roll=172.32.5.231
p3_metaservice=172.32.5.231
p3_egg=172.32.5.231

# services for 4
p4_mysql=172.32.30.240
p4_redis=172.32.30.240
p4_fate_flow=172.32.30.240
p4_fateboard=172.32.30.240
p4_federation=172.32.30.240
p4_proxy=172.32.30.240
p4_roll=172.32.30.240
p4_metaservice=172.32.30.240
p4_egg=172.32.30.240

# services for 5
p5_mysql=172.32.41.41
p5_redis=172.32.41.41
p5_fate_flow=172.32.41.41
p5_fateboard=172.32.41.41
p5_federation=172.32.41.41
p5_proxy=172.32.41.41
p5_roll=172.32.41.41
p5_metaservice=172.32.41.41
p5_egg=172.32.41.41

# services for 6
p6_mysql=172.32.40.104
p6_redis=172.32.40.104
p6_fate_flow=172.32.40.104
p6_fateboard=172.32.40.104
p6_federation=172.32.40.104
p6_proxy=172.32.40.104
p6_roll=172.32.40.104
p6_metaservice=172.32.40.104
p6_egg=172.32.40.104

# services for 7
p7_mysql=172.32.76.236
p7_redis=172.32.76.236
p7_fate_flow=172.32.76.236
p7_fateboard=172.32.76.236
p7_federation=172.32.76.236
p7_proxy=172.32.76.236
p7_roll=172.32.76.236
p7_metaservice=172.32.76.236
p7_egg=172.32.76.236

# services for 8
p8_mysql=172.32.75.125
p8_redis=172.32.75.125
p8_fate_flow=172.32.75.125
p8_fateboard=172.32.75.125
p8_federation=172.32.75.125
p8_proxy=172.32.75.125
p8_roll=172.32.75.125
p8_metaservice=172.32.75.125
p8_egg=172.32.75.125

# services for 9
p9_mysql=172.32.138.4
p9_redis=172.32.138.4
p9_fate_flow=172.32.138.4
p9_fateboard=172.32.138.4
p9_federation=172.32.138.4
p9_proxy=172.32.138.4
p9_roll=172.32.138.4
p9_metaservice=172.32.138.4
p9_egg=172.32.138.4

# services for 10
p10_mysql=172.32.131.35
p10_redis=172.32.131.35
p10_fate_flow=172.32.131.35
p10_fateboard=172.32.131.35
p10_federation=172.32.131.35
p10_proxy=172.32.131.35
p10_roll=172.32.131.35
p10_metaservice=172.32.131.35
p10_egg=172.32.131.35

party_list=(0 1 2 3 4 5 6 7 8 9 10)
party_names=(p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10)