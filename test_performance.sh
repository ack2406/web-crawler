#!/bin/bash

START_URL="https://www.osu.edu"
MAX_PAGES_TEST=1000
CONCURRENCY_LEVELS=(1 2 4 8 16 32 64 128 256 512 1024 2048)
OUTPUT_BASE_DIR="./artifacts_perf_test"
PERFORMANCE_LOG="${OUTPUT_BASE_DIR}/performance_results.csv"
CRAWLER_SCRIPT="crawler.py"

if [ ! -f "$CRAWLER_SCRIPT" ]; then
    echo "Error: Crawler script '$CRAWLER_SCRIPT' not found."
    exit 1
fi

mkdir -p "$OUTPUT_BASE_DIR"

echo "Starting performance test for $CRAWLER_SCRIPT..."
echo "Target URL: $START_URL, Max pages: $MAX_PAGES_TEST"
echo "Concurrency levels: ${CONCURRENCY_LEVELS[@]}"
echo "Results will be saved to: $PERFORMANCE_LOG"
echo "Detailed logs in subdirectories under: $OUTPUT_BASE_DIR"

echo "Concurrency,ExecutionTimeSeconds" > "$PERFORMANCE_LOG"

for CONC in "${CONCURRENCY_LEVELS[@]}"; do
    echo "Running test - Concurrency: $CONC"
    TEST_OUTPUT_DIR="${OUTPUT_BASE_DIR}/concurrency_${CONC}"
    rm -rf "$TEST_OUTPUT_DIR"
    mkdir -p "$TEST_OUTPUT_DIR"

    TIME_OUTPUT_FILE=$(mktemp)

    (time -p python "$CRAWLER_SCRIPT" "$START_URL" -m "$MAX_PAGES_TEST" -c "$CONC" -o "$TEST_OUTPUT_DIR") > "${TEST_OUTPUT_DIR}/crawler_stdout.log" 2> "$TIME_OUTPUT_FILE"
    EXIT_CODE=$?

    REAL_TIME=$(grep '^real' "$TIME_OUTPUT_FILE" | awk '{print $2}')

    rm -f "$TIME_OUTPUT_FILE"

    if [ $EXIT_CODE -eq 0 ] && [ -n "$REAL_TIME" ]; then
        echo "$CONC,$REAL_TIME" >> "$PERFORMANCE_LOG"
        echo "  Finished Concurrency $CONC. Time: ${REAL_TIME}s"
    else
        echo "$CONC,ERROR" >> "$PERFORMANCE_LOG"
        echo "  Error during test for Concurrency $CONC (Exit Code: $EXIT_CODE). Check logs in: $TEST_OUTPUT_DIR"
    fi
done

echo "Performance testing finished."
echo "CSV results saved to: $PERFORMANCE_LOG"

exit 0