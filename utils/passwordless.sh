function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}

if [ ! -f ./id_rsa ]; then
  parse_yaml ssh_conf.yml > temp.sh
  source temp.sh
  rm temp.sh
  ssh-keygen -t rsa -b 4096 -C "" -f ./id_rsa -N ""
fi

pub=`cat id_rsa.pub`
local_pub=`cat ~/.ssh/id_rsa.pub`

python extract_public_ip.py
source temp.sh

for ip in "${ips[@]}"
do
  ssh -o StrictHostKeyChecking=no -l ubuntu ${ip} -i ~/.ssh/MyKeyPair.pem "echo 'ubuntu:ubuntu' | sudo chpasswd"

  # The "sudo" immediately after "chpasswd" does not require passwd! Utilize it...
  ssh -l ubuntu ${ip} -i ~/.ssh/MyKeyPair.pem "sudo sed -i 's|[#]*PasswordAuthentication no|PasswordAuthentication yes|g' /etc/ssh/sshd_config"
  ssh -l ubuntu ${ip} -i ~/.ssh/MyKeyPair.pem "sudo service ssh restart"
  ssh -l ubuntu ${ip} -i ~/.ssh/MyKeyPair.pem "echo ${local_pub} >> ~/.ssh/authorized_keys"
  ssh -l ubuntu ${ip} -i ~/.ssh/MyKeyPair.pem "echo ${pub} >> ~/.ssh/authorized_keys"
  scp -q -i ~/.ssh/MyKeyPair.pem id_rsa ubuntu@${ip}:/home/ubuntu/.ssh/id_rsa
  ssh -l ubuntu ${ip} -i ~/.ssh/MyKeyPair.pem "chmod 600 /home/ubuntu/.ssh/id_rsa"
  scp -q -i ~/.ssh/MyKeyPair.pem id_rsa.pub ubuntu@${ip}:/home/ubuntu/.ssh/id_rsa.pub
  ssh ubuntu@${ip} "echo -e 'Host *\n    StrictHostKeyChecking no' > ~/.ssh/config"
done

rm temp.sh