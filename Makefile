-include .env
export

pull:
	@mkdir -p $(LOCAL_PATH)
	rsync -avz -e "ssh -p $(REMOTE_PORT)" $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_PATH)/ $(LOCAL_PATH)/
	@echo "Pulled submissions to $(LOCAL_PATH)/"

push:
	rsync -avz -e "ssh -p $(REMOTE_PORT)" $(LOCAL_PATH)/ $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_PATH)/
	@echo "Pushed submissions to $(REMOTE_HOST):$(REMOTE_PATH)/"

list-remote:
	ssh -p $(REMOTE_PORT) $(REMOTE_USER)@$(REMOTE_HOST) "ls -lt $(REMOTE_PATH)/"
