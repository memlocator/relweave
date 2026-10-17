#!/usr/bin/env bash
# Train and evaluate relweave on your own schema: steps 5-8 of docs/recipe-own-schema.md in one run, on any machine
# with a GPU (your own, a cloud box, a Hugging Face job). Run from the domain directory that holds the schema module and
# the labelled chunk files gold_train/train.jsonl, gold_val/test.jsonl and gold_test/test.jsonl.
#
#   bash path/to/train_domain.sh mydomain.py:MYSCHEMA
#
# Settings (environment): EPOCHS=3, LR=1e-4, MAX_PROBES=200 (head probes per chunk; about 80 fit an 8 GB card),
# HEAD=1 (0 skips the head: the generator alone, about half the run time), BASE=chrullis/relweave-4b-base,
# ENGINE=transformers (vllm: generation on vLLM, several times faster), RELWEAVE_GEN=relweave (the relweave command
# that generates: with ENGINE=vllm, the one in an environment with vllm and vllm-bnb-plugin, which pin their own torch;
# training stays in the environment with relweave[train]).
# Rerunning continues where a run stopped: a step whose output exists is skipped (delete it to redo the step).
# Writes runs/: the generator (gen/), head (head/) and entity tagger (tagger/), outputs on validation and test, scores, one log per step,
# and runs/summary.txt with the validation scores that decide between the generator alone and the union.
set -uo pipefail
SCHEMA=${1:?usage: train_domain.sh module.py:NAME}
EPOCHS=${EPOCHS:-3} LR=${LR:-1e-4} MAX_PROBES=${MAX_PROBES:-200} HEAD=${HEAD:-1} BASE=${BASE:-chrullis/relweave-4b-base}
ENGINE=${ENGINE:-transformers} RELWEAVE_GEN=${RELWEAVE_GEN:-relweave}
S="--schema $SCHEMA"
for f in gold_train/train.jsonl gold_val/test.jsonl gold_test/test.jsonl; do
  [ -s "$f" ] || { echo "missing $f (recipe-own-schema, steps 3-4)"; exit 1; }
done
mkdir -p runs
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a runs/summary.txt; }
step() { local n=$1; shift; log "step $n"; "$@" > "runs/$n.log" 2>&1; local st=$?
  [ $st -eq 0 ] || { log "step $n FAILED (exit $st), last lines of runs/$n.log:"; tail -5 "runs/$n.log" | tee -a runs/summary.txt; }
  return $st; }
# a step whose output exists is done (outputs are written when a step ends): a crashed run restarts where it stopped
once() { local out=$1; shift; if [ -s "$out" ]; then log "step $1: done before ($out), skipped"; return 0; fi; step "$@"; }

[ -d models/base/generator ] || step download hf download "$BASE" --local-dir models/base || exit 1
once runs/gen/adapter/adapter_config.json train-generator relweave train-generator gold_train/train.jsonl $S --init-adapter models/base/generator \
  --lr "$LR" --epochs "$EPOCHS" --conditioned --epoch-adapters --out runs/gen || exit 1
for s in val test; do
  once runs/$s.jsonl generate-$s "$RELWEAVE_GEN" generate gold_$s/test.jsonl $S --generator runs/gen --out runs/$s.jsonl --engine "$ENGINE" || exit 1
done
log "validation, generator alone:"
relweave eval gold_val/test.jsonl runs/val.jsonl $S --by-type 2>/dev/null | tee -a runs/summary.txt

if [ "$HEAD" = 1 ]; then
  n=$(grep -c . gold_train/train.jsonl)
  if [ ! -s runs/own.jsonl ] && [ "$ENGINE" = transformers ]; then  # vLLM: a 5-record timing is mostly its start-up
    t0=$(date +%s); head -5 gold_train/train.jsonl > runs/time5.jsonl
    "$RELWEAVE_GEN" generate runs/time5.jsonl $S --generator runs/gen --out runs/time5_out.jsonl --engine "$ENGINE" > /dev/null 2>&1
    per=$(( ($(date +%s) - t0) / 5 ))
    log "the head needs the generator on all $n training records: about $(( per * n / 60 )) min at ${per}s per record"
  fi
  once runs/own.jsonl generate-train "$RELWEAVE_GEN" generate gold_train/train.jsonl $S --generator runs/gen --out runs/own.jsonl --engine "$ENGINE" || exit 1
  once runs/head/head_config.json train-head relweave train-head gold_train/train.jsonl $S --generator runs/gen --own-results runs/own.jsonl \
    --max-probes "$MAX_PROBES" --out runs/head || exit 1
  once runs/tagger/tagger.pt train-tagger relweave train-tagger gold_train/train.jsonl $S --head runs/head --out runs/tagger || true
  for s in val test; do
    once runs/${s}_scores.json score-$s relweave score gold_$s/test.jsonl runs/$s.jsonl $S --generator runs/gen --head runs/head \
      --out runs/${s}_scores.json || exit 1
  done
  log "validation, generator + head (union) by cut; keep the union only if a cut beats the generator alone:"
  for cut in -12 -6 -4 -2 0; do
    echo -n "cut $cut: " | tee -a runs/summary.txt
    relweave eval gold_val/test.jsonl runs/val.jsonl $S --scores runs/val_scores.json --cut=$cut --margin=0.5 2>/dev/null \
      | grep union | tee -a runs/summary.txt
  done
fi
log "done. Next: score the test set by pooled judging (docs/quickstart-new-domain.md, step 6)"
