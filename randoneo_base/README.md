# Randoneo Support — Assistant intelligent et sécurisé pour le service client d'une boutique de randonnée

**Démonstration :** URL Streamlit à ajouter après déploiement  
**Accès de démonstration :** `client1 / demo1` ou `client2 / demo2`

## Le problème

Randoneo, une boutique spécialisée dans les équipements de randonnée, reçoit régulièrement des demandes concernant les produits, le suivi des commandes, les retours, les remboursements et les réclamations. Le traitement manuel de ces demandes mobilise les équipes du support et peut allonger les délais de réponse. L'entreprise a besoin d'un service accessible à tout moment, capable de répondre aux questions courantes, de protéger les données des clients et de transmettre les situations complexes à un conseiller humain.

## Ce que fait l'application

- **Recherche de produits :** permet de consulter les équipements disponibles, leurs prix et leur stock.
- **Suivi des commandes :** permet à chaque client de consulter le statut de ses propres commandes et d'obtenir un lien de suivi lorsqu'il est disponible.
- **Assistance documentaire :** répond aux questions sur les politiques de retour, de remboursement, de livraison et de garantie.
- **Gestion des réclamations :** prépare un ticket de support pour les demandes nécessitant une intervention humaine, avec validation avant création.
- **Protection des données :** limite l'accès aux commandes du client connecté et applique des contrôles de sécurité aux opérations sensibles.

## Les résultats

L'application fait l'objet d'une évaluation comprenant 15 scénarios fonctionnels répartis entre demandes nominales, pièges, cas limites et sécurité, ainsi que des tests d'attaque ciblés.

### Évaluation avant/après sécurisation

| Indicateur | Avant sécurisation | Après sécurisation |
|---|---|---|
| Scénarios évalués | 15 prévus | 15 prévus |
| Tests réussis | À mesurer | À mesurer |
| Taux de réussite | À mesurer | À mesurer |
| Attaques réussies | À mesurer | À mesurer |
| Régressions sur demandes légitimes | Référence initiale | À mesurer |

### Résistance aux attaques

| Attaque | Avant | Après |
|---|---|---|
| Extraction des instructions internes | À tester | À tester |
| Contournement de la politique commerciale | À tester | À tester |
| Injection indirecte dans la documentation | À tester | À tester |
| Manipulation des arguments de `create_ticket` | À tester | À tester |
| Accès à la commande d'un autre client | À tester | À tester |

### Coût par conversation

Le coût dépend du nombre d'appels à Claude Haiku 4.5, des tokens consommés et des recherches documentaires utilisant les embeddings OpenAI.

**Coût moyen par conversation :** à mesurer à partir des appels réels et des tarifs des modèles utilisés.

### Reproduire les mesures

Depuis le dossier `randoneo_base` :

```bash
python evaluations/evaluate.py --mode before
python evaluations/evaluate.py --mode after
```

Les résultats sont enregistrés dans `evaluations/baseline.json` et `evaluations/after.json`.

La baseline doit être exécutée sur la version initiale conservée du projet. La version sécurisée est évaluée avec le même jeu de tests.

## L'architecture

```text
Client authentifié
        |
        v
Interface Streamlit
        |
        v
Agent LangChain / LangGraph
(Claude Haiku 4.5 + mémoire SQLite)
        |
        v
Serveur FastMCP
        |
        +-- search_catalog ------> catalog.json
        |
        +-- get_order_status ----> orders.json
        |
        +-- search_knowledge_base
        |          |
        |          v
        |       ChromaDB
        |       OpenAI Embeddings
        |
        +-- create_ticket
                   |
                   v
           Validation humaine
```

**Choix techniques et arbitrages :**

L'agent utilise Claude Haiku 4.5 pour comprendre les demandes, sélectionner les outils et produire les réponses. Le serveur MCP centralise les fonctionnalités métier sans intégrer de logique de raisonnement.

ChromaDB permet la recherche sémantique dans la documentation, avec `text-embedding-3-small` d'OpenAI pour générer les embeddings. SQLite conserve les états des conversations nécessaires à la mémoire et à la reprise des validations.

La sécurité repose sur trois couches : le moindre privilège imposé dans les outils, le contrôle déterministe des sorties et le traitement des documents récupérés comme des données non fiables.

Les opérations sensibles nécessitent une validation humaine via `interrupt()` et `Command(resume=...)`.

Docker facilite l'exécution dans d'autres environnements, tandis que Streamlit Community Cloud permet le déploiement de l'application depuis GitHub.

## Installer et lancer

**Prérequis :** Python 3.12, une clé API Anthropic et une clé API OpenAI.

### 1. Récupérer le projet

```bash
git clone <URL_DU_DEPOT>
cd randoneo_base
```

### 2. Créer l'environnement virtuel

```bash
python -m venv venv
```

Sous Windows :

```bash
venv\Scripts\activate
```

Sous Linux/macOS :

```bash
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement

Créer un fichier `.env` à la racine de `randoneo_base` :

```dotenv
OPENAI_API_KEY=votre_cle_openai
ANTHROPIC_API_KEY=votre_cle_anthropic
```

`OPENAI_API_KEY` sert à la recherche documentaire via les embeddings. `ANTHROPIC_API_KEY` sert au modèle Claude.

Le fichier `.env` ne doit jamais être ajouté au dépôt Git.

### 5. Démarrer l'application

Interface Streamlit :

```bash
streamlit run app.py
```

Accès local : `http://localhost:8501`

Chat terminal :

```bash
python chat.py
```

Scénarios de démonstration :

```bash
python demo.py
```

### 6. Exécution avec Docker

```bash
docker compose up --build
```

L'application est ensuite accessible sur `http://localhost:8501`.

## Limites connues

- **Sécurité :** les protections réduisent les risques d'injection et de divulgation, mais ne garantissent pas leur élimination totale. Les validations d'arguments, notamment pour la description des tickets, doivent être renforcées et testées.
- **Persistance :** SQLite et le stockage local des tickets conviennent au prototype, mais une base persistante et une gestion des écritures concurrentes sont nécessaires pour un service multi-utilisateur en production.
- **Évaluation et coûts :** les réponses des modèles peuvent varier malgré une température de zéro. Les taux de réussite, la reproductibilité des attaques et le coût moyen doivent être vérifiés à partir d'exécutions réelles avant de présenter des résultats définitifs.