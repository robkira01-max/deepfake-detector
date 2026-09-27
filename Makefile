# DeepfakeDetector Canada — Makefile
# Usage: make help

.PHONY: help dev test coverage demo up down logs shell migrate lint bandit transfer

SHELL := /bin/bash
VENV  := source .venv/bin/activate
BACK  := backend

# ── Aide ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  DeepfakeDetector Canada — Commandes"
	@echo "  ════════════════════════════════════"
	@echo ""
	@echo "  Développement local :"
	@echo "    make dev        — Lance API + worker + demo (3 terminaux)"
	@echo "    make api        — API FastAPI seule (port 8001)"
	@echo "    make worker     — Celery worker seul"
	@echo "    make demo       — Serveur demo investisseurs (port 8080)"
	@echo ""
	@echo "  Tests :"
	@echo "    make test             — Suite unitaire (548+ tests, skip integration)"
	@echo "    make test-fast        — Suite unitaire rapide (stop au 1er echec)"
	@echo "    make test-integration — Tests integration PostgreSQL+Redis (Docker)"
	@echo "    make test-all         — Tous les tests (unitaires + integration)"
	@echo "    make coverage         — Tests + rapport couverture HTML"
	@echo "    make lint             — Verification types (mypy)"
	@echo "    make bandit           — Scan SAST securite"
	@echo ""
	@echo "  Docker :"
	@echo "    make up         — Lance tous les services Docker"
	@echo "    make down       — Arrête tous les services"
	@echo "    make logs       — Logs en temps réel"
	@echo "    make build      — Rebuild les images"
	@echo "    make ps         — État des services"
	@echo ""
	@echo "  Base de données :"
	@echo "    make migrate    — Applique les migrations Alembic"
	@echo "    make migration  — Crée une nouvelle migration (MSG=...)"
	@echo ""
	@echo "  Utilitaires :"
	@echo "    make demo-cli   — Demo CLI sur fichier test"
	@echo "    make transfer   — Copie vers Windows share"
	@echo "    make clean      — Supprime __pycache__, .coverage, etc."
	@echo ""

# ── Développement local ───────────────────────────────────────────────────────
api:
	@$(VENV) && cd $(BACK) && uvicorn main:app --host 127.0.0.1 --port 8001 --reload --log-level info

worker:
	@$(VENV) && PYTHONPATH=$(PWD)/$(BACK) celery -A tasks.analysis_tasks worker \
	  --loglevel=info --concurrency=1

demo:
	@python3 -m http.server 8080 --directory demo/

dev:
	@echo "Lance 3 terminaux séparés avec : make api | make worker | make demo"
	@echo "Ou utilisez tmux : tmux new-session 'make api' \; split-window 'make worker' \; split-window 'make demo'"

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	@$(VENV) && cd $(BACK) && python -m pytest tests/ -q --tb=short -m "not integration"

test-fast:
	@$(VENV) && cd $(BACK) && python -m pytest tests/ -q --tb=short -x -m "not integration"

test-integration:
	@echo "Tests d integration (PostgreSQL + Redis via Docker)..."
	@$(VENV) && cd $(BACK) && python -m pytest tests/test_integration.py -v -m integration

test-all:
	@echo "Suite complete (unitaires + integration)..."
	@$(VENV) && cd $(BACK) && python -m pytest tests/ -v --tb=short

coverage:
	@$(VENV) && cd $(BACK) && python -m pytest tests/ \
	  --cov=engines --cov=core --cov=routers \
	  --cov-report=term-missing \
	  --cov-report=html:../docs/coverage \
	  -q --tb=no
	@echo "Rapport HTML : docs/coverage/index.html"

lint:
	@$(VENV) && cd $(BACK) && python -m mypy . --ignore-missing-imports \
	  --exclude 'tests|\.venv|__pycache__' 2>&1 | tail -20

bandit:
	@$(VENV) && bandit -r $(BACK)/ -x $(BACK)/tests/,$(BACK)/.venv/ -ll \
	  -f txt -o docs/security/bandit-report.txt
	@echo "Rapport : docs/security/bandit-report.txt"
	@$(VENV) && bandit -r $(BACK)/ -x $(BACK)/tests/,$(BACK)/.venv/ -ll 2>&1 | tail -10

# ── Docker ────────────────────────────────────────────────────────────────────
up:
	@docker compose up -d
	@echo ""
	@echo "  Services disponibles :"
	@echo "    API FastAPI  : http://localhost/docs"
	@echo "    API health   : http://localhost/health"
	@echo "    Flower       : http://localhost:5555"
	@echo "    MinIO        : http://localhost:9001"
	@echo ""

down:
	@docker compose down

build:
	@docker compose build --no-cache

logs:
	@docker compose logs -f --tail=50

ps:
	@docker compose ps

restart-api:
	@docker compose restart api

shell:
	@docker compose exec api bash

# ── Base de données ───────────────────────────────────────────────────────────
migrate:
	@$(VENV) && cd $(BACK) && alembic upgrade head

migration:
	@[ -n "$(MSG)" ] || (echo "Usage: make migration MSG='description'" && exit 1)
	@$(VENV) && cd $(BACK) && alembic revision --autogenerate -m "$(MSG)"

db-reset:
	@$(VENV) && cd $(BACK) && alembic downgrade base && alembic upgrade head

# ── Utilitaires ───────────────────────────────────────────────────────────────
demo-cli:
	@$(VENV) && python -c "
import wave, struct
with wave.open('/tmp/deepfake_demo.wav', 'w') as f:
    f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
    f.writeframes(struct.pack('<' + 'h'*3200, *([0]*3200)))
" && $(VENV) && python demo.py /tmp/deepfake_demo.wav --verbose

transfer:
	@rsync -av \
	  --exclude='__pycache__' --exclude='.venv' --exclude='.git' \
	  --exclude='*.pyc' --exclude='.coverage' --exclude='keys/' \
	  --exclude='data/' --exclude='*.db' \
	  /home/kali/deepfake_detector/ \
	  /media/sf_Kali-share/DeepfakeDetector/source/
	@echo "Transfert vers Windows terminé."

clean:
	@find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null; true
	@find . -name '*.pyc' -delete 2>/dev/null; true
	@rm -f $(BACK)/.coverage
	@echo "Nettoyage terminé."

keys-gen:
	@$(VENV) && cd $(BACK) && python -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path
from core.security import generate_rsa_keypair
generate_rsa_keypair(Path('keys'))
print('Clés RSA-4096 générées dans backend/keys/')
"
