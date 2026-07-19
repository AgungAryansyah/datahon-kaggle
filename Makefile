-include .env
export

REMOTE_PORT ?= 22
SSH_CMD := ssh -p $(REMOTE_PORT)
ifeq ($(REMOTE_PORT),)
SSH_CMD := ssh
endif

RSYNC_SSH := $(SSH_CMD)

NB_DIR := task2/notebooks

.PHONY: sync run-nb run-all pull-submissions ssh push pull

sync:
	rsync -avz -e "$(RSYNC_SSH)" \
		--exclude '.venv' --exclude '.cache' --exclude '__pycache__' \
		--exclude '*.pyc' --exclude '.ipynb_checkpoints' \
		--exclude 'dataset-task1' --exclude 'dataset-task2/screenshots' \
		--exclude 'dataset-task2/*.csv' --exclude 'dataset-task2/*.zip' \
		--exclude 'submissions' \
		./ $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_ROOT)/
	@echo "Synced to $(REMOTE_HOST):$(REMOTE_ROOT)/"

run-nb:
	@test -n "$(name)" || { echo "Usage: make run-nb name=<notebook_name>"; exit 1; }
	$(SSH_CMD) $(REMOTE_USER)@$(REMOTE_HOST) \
		"cd $(REMOTE_ROOT) && uv run jupyter nbconvert --to notebook --execute $(NB_DIR)/$(name).ipynb --output $(NB_DIR)/$(name).ipynb --inplace"
	@echo "Ran $(name).ipynb on $(REMOTE_HOST)"

run-all:
	$(SSH_CMD) $(REMOTE_USER)@$(REMOTE_HOST) \
		"cd $(REMOTE_ROOT) && uv run jupyter nbconvert --to notebook --execute $(NB_DIR)/01_eda.ipynb --output $(NB_DIR)/01_eda.ipynb --inplace"
	$(SSH_CMD) $(REMOTE_USER)@$(REMOTE_HOST) \
		"cd $(REMOTE_ROOT) && uv run jupyter nbconvert --to notebook --execute $(NB_DIR)/02_baseline.ipynb --output $(NB_DIR)/02_baseline.ipynb --inplace"
	$(SSH_CMD) $(REMOTE_USER)@$(REMOTE_HOST) \
		"cd $(REMOTE_ROOT) && uv run jupyter nbconvert --to notebook --execute $(NB_DIR)/03_deep_learning.ipynb --output $(NB_DIR)/03_deep_learning.ipynb --inplace"
	@echo "All notebooks executed on $(REMOTE_HOST)"

pull-submissions:
	rsync -avz -e "$(RSYNC_SSH)" $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_ROOT)/submissions/ ./submissions/ 2>/dev/null || true
	rsync -avz -e "$(RSYNC_SSH)" $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_ROOT)/task2/notebooks/submissions/ ./submissions/ 2>/dev/null || true
	@echo "Pulled submissions to ./submissions/"

ssh:
	$(SSH_CMD) $(REMOTE_USER)@$(REMOTE_HOST)

push:
	rsync -avz -e "$(RSYNC_SSH)" $(LOCAL_PATH)/ $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_PATH)/
	@echo "Pushed $(LOCAL_PATH) -> $(REMOTE_HOST):$(REMOTE_PATH)/"

pull:
	rsync -avz -e "$(RSYNC_SSH)" $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_PATH)/ $(LOCAL_PATH)/
	@echo "Pulled $(REMOTE_HOST):$(REMOTE_PATH) -> $(LOCAL_PATH)/"
