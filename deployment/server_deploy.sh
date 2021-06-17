CONFIG_FILE=$1

# These are necessary to accommodate FATE's CentOS scripts
# when you are running atop Ubuntu
adapt() {
  echo "[INFO] Adapt to Ubuntu..."
  if ! command -v java &> /dev/null; then
    wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/jdk-8u192-linux-x64.tar.gz
    sudo mkdir -p /usr/lib/jvm/
    sudo tar xzf jdk-8u192-linux-x64.tar.gz -C /usr/lib/jvm/
    JAVA_HOME=/usr/lib/jvm/jdk1.8.0_192/
    sudo update-alternatives --install /usr/bin/java java ${JAVA_HOME%*/}/bin/java 20000
    sudo update-alternatives --install /usr/bin/javac javac ${JAVA_HOME%*/}/bin/javac 20000
  fi

  if ! command -v mvn &> /dev/null; then
    wget https://www-us.apache.org/dist/maven/maven-3/3.6.3/binaries/apache-maven-3.6.3-bin.tar.gz -P /tmp
    sudo tar xf /tmp/apache-maven-*.tar.gz -C /opt
    sudo ln -s /opt/apache-maven-3.6.3 /opt/maven
    sudo vim /etc/profile.d/maven.sh
    export JAVA_HOME=/usr/lib/jvm/jdk1.8.0_192
    export M2_HOME=/opt/maven
    export MAVEN_HOME=/opt/maven
    export PATH=${M2_HOME}/bin:${PATH}
    sudo chmod +x /etc/profile.d/maven.sh
    source /etc/profile.d/maven.sh
  fi
}

# Update FATE's configuration files
config() {
  echo "[INFO] Updating FATE's configuration files..."
  python modify_fate_configs.py ${CONFIG_FILE}
}

# Package FATE
package() {
  echo "[INFO] Packaging..."
  cd ../cluster-deploy/scripts/
  bash packaging_1.sh
  cd ../../
  cp utils/RuntimeUtils.java eggroll/core/src/main/java/com/webank/ai/eggroll/core/utils/
  cd cluster-deploy/scripts/
  bash packaging_2.sh
}

# deploy FATE
deploy() {
  echo "[INFO] Deploying..."
  source ./multinode_cluster_configurations.sh
  bash deploy_cluster_multinode.sh build all
}

adapt
config
package
deploy