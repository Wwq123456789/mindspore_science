#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

SCRIPT_BASEDIR=$(realpath "$(dirname "$0")")

PROJECT_DIR=$(realpath "$SCRIPT_BASEDIR/../../")
ST_PATH="$PROJECT_DIR/tests/st"

if [ $# -gt 0 ]; then
  if  [ $1 == "mindelec" ]; then
    echo "Run st mindelec."
    cd "$PROJECT_DIR" || exit
    ST_PATH="$PROJECT_DIR/tests/st/mindelec/"
    pytest "$ST_PATH"
    echo "Test all mindelec use cases success."
  elif [ $1 == "mindsponge" ]; then
    echo "Run st mindsponge."
    cd "$PROJECT_DIR" || exit
    ST_PATH="$PROJECT_DIR/tests/st/mindsponge/"
    pytest "$ST_PATH"
    echo "Test all mindsponge use cases success."
  elif [ $1 == "mindflow" ]; then
    echo "Run st mindflow."
    cd "$PROJECT_DIR" || exit
    ST_PATH="$PROJECT_DIR/tests/st/mindflow/"
    pytest "$ST_PATH"
    echo "Test all mindflow use cases success."
  fi
else
  echo "Run all st."
  cd "$PROJECT_DIR" || exit
  ST_PATH="$PROJECT_DIR/tests/st/"
  pytest "$ST_PATH"
  echo "Test all use cases success."
  fi
