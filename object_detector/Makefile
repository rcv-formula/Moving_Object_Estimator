# ====== Compose files ======
COMPOSE_NOGPU := compose/compose.yaml
COMPOSE_GPU   := compose/compose.desktop.yaml
COMPOSE_JETSON:= compose/compose.jetson.yaml

# ====== Container names ======
C_NOGPU  := mos_desktop
C_GPU    := mos_desktop_gpu
C_JETSON := mos_jetson

# ====== FLAGS ======
COMPOSE_FLAGS := --project-directory .

# ====== Helpers ======
define _exec_or_hint
	@cid=$$(docker ps -q -f name=$(1)); \
	if [ -z "$$cid" ]; then \
		echo "컨테이너 '$(1)'가 실행 중이 아닙니다. 먼저 'make $(2)' 로 올려주세요."; \
		exit 1; \
	fi; \
	docker exec -it $(1) bash
endef

define _logs_or_hint
	@cid=$$(docker ps -q -f name=$(1)); \
	if [ -z "$$cid" ]; then \
		echo "컨테이너 '$(1)'가 실행 중이 아닙니다. 먼저 'make $(2)' 로 올려주세요."; \
		exit 1; \
	fi; \
	docker logs -f --tail=200 $(1)
endef

# ====== Up (빌드 자동 포함) ======
up-nogpu:
	docker compose $(COMPOSE_FLAGS) -f $(COMPOSE_NOGPU) up -d --build

up-gpu:
	docker compose $(COMPOSE_FLAGS) -f $(COMPOSE_GPU) up -d --build

up-jetson:
	docker compose $(COMPOSE_FLAGS) -f $(COMPOSE_JETSON) up -d --build

# ====== Exec (컨테이너 셸 진입) ======
exec-nogpu:
	$(call _exec_or_hint,$(C_NOGPU),up-nogpu)

exec-gpu:
	$(call _exec_or_hint,$(C_GPU),up-gpu)

exec-jetson:
	$(call _exec_or_hint,$(C_JETSON),up-jetson)

# ====== Logs ======
logs-nogpu:
	$(call _logs_or_hint,$(C_NOGPU),up-nogpu)

logs-gpu:
	$(call _logs_or_hint,$(C_GPU),up-gpu)

logs-jetson:
	$(call _logs_or_hint,$(C_JETSON),up-jetson)

# ====== Misc ======
ps:
	docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"

down:
	docker compose $(COMPOSE_FLAGS) -f $(COMPOSE_NOGPU) down || true
	docker compose $(COMPOSE_FLAGS) -f $(COMPOSE_GPU) down || true
	docker compose $(COMPOSE_FLAGS) -f $(COMPOSE_JETSON) down || true

clean:
	docker system prune -af --volumes

help:
	@echo "make up-nogpu     # Non-GPU 환경: 빌드+실행"
	@echo "make up-gpu       # GPU 환경: 빌드+실행"
	@echo "make up-jetson    # Jetson: 빌드+실행"
	@echo "make exec-nogpu   # Non-GPU 컨테이너 bash 접속 ($(C_NOGPU))"
	@echo "make exec-gpu     # GPU 컨테이너 bash 접속 ($(C_GPU))"
	@echo "make exec-jetson  # Jetson 컨테이너 bash 접속 ($(C_JETSON))"
	@echo "make logs-gpu     # GPU 컨테이너 로그 팔로우"
	@echo "make ps           # 실행 중 컨테이너 목록"
	@echo "make down         # 모든 compose 스택 종료"
	@echo "make clean        # 이미지/컨테이너/볼륨 강제 정리(주의)"
