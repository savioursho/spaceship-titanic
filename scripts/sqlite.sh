#!/bin/bash

source .env

sqlite3 \
    -cmd "ATTACH DATABASE '${SQLITE_DB_DIR}/train.db' as train;" \
    -cmd "ATTACH DATABASE '${SQLITE_DB_DIR}/test.db' as test;" \
    ${SQLITE_DB_DIR}/main.db 
