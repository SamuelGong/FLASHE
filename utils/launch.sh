#!/bin/bash
pip install -q boto3 ruamel.yaml
python batch_launch.py cluster_plan.yml instance_template.yml

# these are flashe-related logics
# you can comment them for general use
python flashe_related.py allow_ingress
python flashe_related.py update_flashe_config