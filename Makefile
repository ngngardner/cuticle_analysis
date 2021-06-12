build:
	bash scripts/build.sh
	bash scripts/test.sh

test:
	bash scripts/test.sh

run:
	bash scripts/run.sh

clean:
	rm -rf logs output dataset
