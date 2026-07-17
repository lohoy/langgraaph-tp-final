# TP Final — Assistant Développeur LangGraph

## Objectif

Construire un assistant développeur LangGraph : l'agent cherche dans la documentation officielle (RAG), propose du code, et demande validation humaine avant d'exécuter quoi que ce soit.

Deux outils sont fournis, prêts à l'emploi :

- `search_docs` — recherche hybride (BM25 + embeddings) dans `docs/`
- `run_python` — exécution de code Python en sous-processus, avec timeout

Votre travail est le **graphe** : état, routage, validation humaine, boucle.

## Installation

```bash
uv sync
uv pip install -e .
```

`search_docs` tourne en local (embeddings + BM25), aucune clé API requise pour ces outils. Ajoutez votre propre configuration pour le modèle de l'agent (selon votre fournisseur).

## Usage

```python
from agent_tools.tools import search_docs, run_python

tools = [search_docs, run_python]
```

## Étapes

1. Définir l'état : `AgentState(MessagesState)` avec un champ `validated`, puis lier les outils au modèle avec `bind_tools`
2. Router par nom d'outil : un appel sensible dans le lot → `validate` ; sinon → `tools` ; rien → `END`
3. Construire le point d'approbation : `validate_node` avec `interrupt()`, `approval_route` vers `tools` ou `reject`
4. Fermer le refus proprement : `reject_node` qui renvoie un `ToolMessage` portant le `tool_call_id`
5. Refermer la boucle : arête `tools` → `agent`, compiler avec `MemorySaver`
6. Activer le streaming (`stream_mode="values"`) et le tracing LangSmith (`LANGSMITH_TRACING=true` + clé)
7. Tester les deux chemins : une question qui déclenche une recherche, une qui déclenche une exécution de code — approuvée puis refusée
8. Documenter : README avec installation, configuration, exemple d'utilisation

## Critères d'acceptation

- **Correct** : le graphe compile, l'agent cherche dans la doc et exécute du code
- **Persistant** : le checkpointer sauvegarde, l'interruption fonctionne, la reprise aussi
- **Traçable** : une trace LangSmith montre le déroulement complet
- **Documenté** : README clair — installation, configuration, exemple d'utilisation

## Rendu

Formulaire de soumission : https://forms.gle/KQ1C932dtEVFReJe9

## Execution

Créer un fichier `.env` à la racine avec au minimum :
```
LANGSMITH_TRACING
LANGSMITH_ENDPOINT
LANGSMITH_API_KEY
LANGSMITH_PROJECT
MISTRAL_API_KEY
```

### 3. LangGraph Studio (recommandé)

```bash
uv run langgraph dev
```

> Requiert un fichier `langgraph.json` à la racine pointant vers le graphe.

Une fois LangSmith Studio ouvert dans le navigateur :
- Se placer sur le chat
- Converser avec l'agent
- Le graphe se met en pause si l'outil run_python est utilisé. Indiquer : "Oui", "oui", "yes", "Yes", "YES", "True", "true", "TRUE", "1" pour valider l'execution

### 4. Test en local

Décommenter les dernières lignes de `src/agent_tools/__init__.py` puis :

```bash
uv run src/agent_tools/__init__.py
```