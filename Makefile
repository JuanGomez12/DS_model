#!make

BLUE="\033[00;94m"
GREEN="\033[00;92m"
RED="\033[00;31m"
RESTORE="\033[0m"
YELLOW="\033[00;93m"
CYAN="\e[0;96m"
GREY="\e[2:N"


test_debug:
	conda run --no--capture--output -n test coverage_run -m pytest
	conda run --no--capture--output -n test coverage html -d /app/test/coverage_report

coverage_report:
	echo ${GREEN} Running unit tests...{RESTORE}
	docker compose run --rm test bash -c "make test_debug"
