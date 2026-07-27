# Voixful — developer conveniences. Real logic lives in scripts/.
.DEFAULT_GOAL := help

.PHONY: help build test release version

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

build: ## Debug build
	swift build

test: ## Run the test suite
	swift test

version: ## Print the current version
	@cat VERSION 2>/dev/null || echo "0.0.0 (unreleased)"

release: ## Interactive release: pick major/minor/patch, bump, tag, push
	@bash scripts/release.sh
