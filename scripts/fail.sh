#!/bin/bash

if [ ! -z "$GRADER_SSH" ]; then
  echo $TRAVIS_REPO_SLUG
  FILENAME=`basename $TRAVIS_REPO_SLUG`
  scp -i ~/.ssh/travislog_rsa $TRAVIS_BUILD_DIR/$FILENAME.log root@199.60.17.67:~/CS431/ASS6/FAIL/
fi
