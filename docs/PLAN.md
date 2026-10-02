# Plan de développement — agent de pentest local

- **Version :** 1.0 — 2 octobre 2026
- **Statut :** feuille de route, non encore exécutée
- **Périmètre :** prototype local, terminal d’abord, destiné exclusivement aux évaluations autorisées.

> Ce plan reprend les décisions d’architecture données dans le contexte de reprise. Lors de l’audit initial, le checkout ne contenait que `README.md` ; depuis, `docs/PLAN.md`, `scripts/diagnostic.py` et `tests/test_diagnostic.py` ont été créés dans cette branche. Les documents d’architecture et le prompt cités dans le contexte ne sont toujours pas présents. Le répertoire du dépôt est pris comme racine du projet.

## 1. Résultat visé et règles qui ne changent pas

La v1 est un agent local qui sait valider un périmètre, faire de la reconnaissance et de l’énumération autorisées, conserver les observations, puis produire un rapport traçable. Il explique ses choix comme un professeur tout en exécutant seul les tâches autorisées dans son périmètre.

Les invariants de conception sont :

- **Traitement local :** inférence et embeddings locaux, sans API de raisonnement ou d’embeddings dans le cloud. Le corpus et les résultats restent sur la machine.
- **Autonomie bornée :** autonomie pour le scope, la recon, l’énumération et le rapport ; arrêt avant toute exploitation ou post-exploitation. Pas de furtivité ni d’évasion de détection.
- **Autorité du code :** le LLM propose ; le code valide et décide. Un prompt ne constitue jamais une barrière de sécurité.
- **Trois portes indépendantes :** autorisation déclarée par un `scope.yaml` valide, destination filtrée hors du conteneur, et preuve vérifiable avant d’inclure une trouvaille au rapport.
- **Séparation cerveau/mains :** le cerveau (modèles, orchestration, base et rapports) n’exécute pas de commande et n’a pas de route réseau vers la cible. Les outils tournent dans des conteneurs jetables, eux-mêmes hébergés dans une VM dédiée.
- **État déterministe :** les faits viennent de parseurs déterministes et d’artefacts conservés, jamais d’une interprétation du LLM seule. Les tentatives infructueuses sont mémorisées pour éviter les répétitions.
- **Deux modèles, chargés par lots :** un orchestrateur fiable en *tool calling* (capacité à demander un outil selon un schéma structuré) et un spécialiste qui ne rend que du texte. Un seul modèle résident à la fois ; aucune bascule à chaque appel.
- **Configuration initiale à mesurer :** 8 Go de VRAM, contexte de 8 192 tokens, cache KV en `q8_0` et paramètres Ollama demandés (`OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_MAX_LOADED_MODELS=1`). Ces valeurs restent à confirmer sur la machine réelle.

**Vocabulaire.** Le *cache KV* (key/value) conserve les représentations des tokens déjà traités afin de poursuivre une génération ; il consomme de la mémoire vidéo. Le *RAG* (génération augmentée par récupération) retrouve des passages d’un corpus local pour fournir du contexte au modèle. Un *Modelfile* est la recette Ollama qui fixe le modèle de base et ses paramètres.

## 2. Vue d’ensemble des jalons

| Jalon | But | Sortie principale | Condition de passage |
|---|---|---|---|
| **M0** | Mesurer la machine et figer les deux modèles locaux | diagnostic, décisions de modèles, Modelfiles | modèles chargés séparément et tests synthétiques concluants |
| **M1** | Rendre le périmètre exécutable et fermé par défaut | schéma et validateur de `scope.yaml` | toute absence ou erreur de scope bloque le démarrage |
| **M2** | Construire la mémoire fiable de l’agent | graphe local, parseurs nmap/ffuf, registre de preuves | mêmes entrées = mêmes faits ; aucune preuve, aucune trouvaille confirmée |
| **M3** | Isoler l’exécution des outils | VM, Compose durci, pare-feu VM | tests démontrant l’absence de route du cerveau et le refus hors scope |
| **M4** | Ajouter la connaissance locale | pipeline RAG local et corpus versionné | index reproductible, sources attribuées, aucune autorité donnée au contenu récupéré |
| **M5** | Relier les composants dans un agent terminal | orchestration par lots, pédagogie, recon/énumération bornées | l’agent reste dans le scope et s’arrête avant les actions à impact |
| **M6** | Valider le système de bout en bout | banc d’essai, rapports de tests et mesures | tous les garde-fous passent ; aucune défaillance ouverte de sécurité |

**Ordre impératif :** aucun outil ne cible un système réel avant la réussite de M1 et M3. M5 ne commence l’exécution intégrée qu’après ces deux portes ; avant cela, les essais utilisent des fixtures et des outils simulés. L’interface web locale n’appartient pas à la v1 : elle viendra après le passage de M6.

## 3. Détail des jalons

### M0 — Point de départ, diagnostic et modèles locaux

**Objectif.** Remplacer les suppositions matérielles par des mesures et vérifier que les deux rôles de modèle tiennent dans la configuration réelle.

**Travail prévu.**

1. Confirmer le modèle exact du GPU et sa VRAM, la RAM, l’OS hôte et l’emplacement de la VM/du moteur Docker. Aucun débit ni budget mémoire ne sera annoncé avant cette mesure.
2. Le script standard library `scripts/diagnostic.py` et ses tests unitaires ont été créés dans ce checkout. Lancer `python scripts/diagnostic.py --json` sur la machine cible pour mesurer GPU/RAM ; le test de débit s’effectuera séparément avec `--bench MODELE`, après installation locale et sélection du modèle.
3. Comparer dans LM Studio les candidats fournis : Qwen3-8B et Cyber-Ornith-1.5-9B pour l’orchestrateur ; WhiteRabbitNeo 2.5 Qwen2.5-Coder-7B et pentest-v2 pour le spécialiste. Consigner les versions, quantifications, licences, formats de prompt et résultats ; ne pas présumer qu’un candidat est meilleur sans le mesurer.
4. Créer les Modelfiles Ollama retenus et configurer une seule résidence de modèle, le contexte et le cache KV conformément aux limites mesurées. Utiliser Ollama en production ; LM Studio reste l’outil de comparaison. La sélection finale doit privilégier la fiabilité réelle du *tool calling* pour l’orchestrateur.
5. Tester sur des scénarios synthétiques, sans cible réelle : choix de l’outil attendu, arguments conformes au schéma et cas piège où aucun appel d’outil n’est justifié. Le spécialiste est évalué comme générateur de texte, pas comme agent d’exécution.

**Livrables.** `scripts/diagnostic.py` et ses tests ; deux Modelfiles retenus ; un manifeste local des modèles et paramètres ; une note de résultats et de décision, y compris les modèles rejetés et la marge VRAM observée.

**Sortie M0.** Diagnostic exécuté et archivé ; chaque modèle tient seul dans la mémoire disponible sans OOM ; résultats des scénarios documentés. Si aucun candidat ne respecte le budget, on réduit le modèle ou les paramètres avant de poursuivre. Aucun modèle cloud ne sert de solution de repli.

### M1 — Contrat de scope et autorisation fail-closed

**Objectif.** Faire du `scope.yaml` une condition de démarrage contrôlée par le code. *Fail-closed* signifie qu’en cas d’absence, d’ambiguïté ou d’erreur, l’accès est refusé.

**Travail prévu.**

- Définir un schéma versionné : identifiant d’engagement ou de labo, période de validité, cibles explicites (IP, CIDR ou noms autorisés), ports/protocoles, exclusions et limites d’exécution. Les jokers globaux et champs inconnus sont refusés.
- Écrire un chargement YAML sûr et strict : clés dupliquées refusées, types contrôlés, aucune coercition permissive. L’absence de fichier, un scope vide, expiré, invalide ou contradictoire empêche la création d’une session et d’un conteneur d’outils.
- Définir la résolution des noms sans faire confiance à une résolution ancienne : vérifier l’adresse effectivement jointe, appliquer les exclusions en priorité, et refuser les destinations non couvertes. Les cas IPv4/IPv6 et les changements de DNS seront testés.
- Garder la validation et les décisions d’autorisation dans le code. Le modèle ne peut ni élargir le scope ni modifier les exclusions.

**Livrables.** Schéma/documentation du scope, validateur, exemple volontairement inactif (aucune cible autorisée par défaut) et tests unitaires couvrant les refus. Le scope déclare l’autorisation opérateur ; le programme ne prétend pas vérifier à lui seul qu’un contrat juridique existe.

**Sortie M1.** Tests prouvant qu’un scope absent ou mal formé bloque avant toute création de processus ou trafic réseau. Aucun repli vers une cible, un domaine ou un port implicite.

### M2 — Graphe de connaissance, parseurs et preuves

**Objectif.** Stocker des faits reproductibles sans demander au LLM de les inventer ou de les interpréter comme des résultats vérifiés.

**Travail prévu.**

- Définir le graphe local : hôtes, services/ports, hypothèses, tentatives, observations et trouvailles, reliés à une session et à leurs sources. Une base locale simple (par exemple SQLite) suffit pour la v1 ; pas de service de graphe distant.
- Implémenter des parseurs déterministes pour les formats machines : XML de nmap et JSON de ffuf. Les sorties terminal destinées aux humains ne seront pas la source de vérité.
- Conserver les artefacts bruts de chaque essai, leur empreinte, l’heure, l’outil/version et le lien avec les faits extraits. Le parseur crée des observations ; une hypothèse reste une hypothèse tant qu’une preuve reproductible ne l’étaye pas.
- Mémoriser les essais et impasses avec leur contexte afin de reconnaître une répétition. Une nouvelle tentative reste possible si le contexte change ; l’agent doit alors expliquer ce changement.
- Séparer le texte explicatif du LLM des champs confirmés du rapport. Un artefact manquant, altéré ou non vérifiable ne peut pas soutenir une trouvaille confirmée.

**Livrables.** Modèle de données et migrations, graphe SQLite, parseurs nmap/ffuf, dépôt local d’artefacts, fixtures synthétiques et tests de répétabilité, doublons, entrées tronquées et erreurs de format.

**Sortie M2.** L’analyse répétée du même artefact produit les mêmes observations ; les faits gardent leur provenance ; une sortie mal formée ou une hypothèse du LLM ne devient jamais une trouvaille vérifiée.

### M3 — VM, conteneurs jetables et pare-feu de destination

**Objectif.** Faire respecter l’isolation par l’infrastructure, pas par le prompt. Ce jalon dépend du choix OS/VM relevé en M0.

**Topologie visée.** Le cerveau reste sur son réseau local sans route vers les cibles. La VM dédiée héberge le contrôleur d’exécution et les conteneurs « mains » jetables. Un canal de contrôle étroit relie les deux ; il ne donne pas au cerveau une interface réseau de cible ni un shell privilégié.

**Travail prévu.**

- Écrire un `docker-compose.yml` pour l’exécution dans la VM. Par défaut : utilisateur non-root, système de fichiers en lecture seule, capacités Linux supprimées au maximum, `no-new-privileges`, limites CPU/mémoire/PID, volumes explicitement limités, aucun mode privilégié et aucun montage du socket Docker dans le cerveau ou les outils.
- Démarrer les conteneurs de travail à la demande, détruire leur environnement après la tâche et ne conserver que les artefacts attendus. Toute permission supplémentaire doit être justifiée par un essai en labo et limitée au strict nécessaire.
- Appliquer un refus réseau par défaut depuis le pare-feu de la VM, hors du conteneur. La chaîne Docker `DOCKER-USER` est le point de contrôle prévu pour les flux des conteneurs ; la politique ne repose ni sur un filtre dans le prompt ni sur la bonne volonté du conteneur.
- Faire vérifier le scope par le composant VM qui applique les règles. Tester les destinations incluses, les exclusions, les refus hors scope, IPv4/IPv6, DNS, connexions déjà établies et comportement après arrêt/redémarrage. Ne jamais installer ces règles sur l’hôte de travail par défaut.
- Documenter une procédure de remise à zéro et de vérification du pare-feu avant chaque campagne. Une erreur de génération ou d’application de règles doit empêcher l’exécution, pas ouvrir le réseau.

**Livrables.** Compose durci, politique et générateur de règles adaptés à la VM, tests réseau confinés à un labo contrôlé, procédure de démarrage/arrêt et journalisation des actions. Le fichier exact dépendra de la plateforme confirmée en M0.

**Sortie M3.** Démonstration que le cerveau ne peut pas joindre directement la cible, qu’un conteneur ne joint que les destinations autorisées et que toute erreur ferme l’accès. Aucun test sur un réseau tiers.

### M4 — RAG entièrement local

**Objectif.** Améliorer la qualité des explications et des pistes sans donner d’autorité de sécurité aux documents récupérés.

**Travail prévu.**

- Construire un pipeline d’ingestion local pour HackTricks, GTFOBins, LOLBAS, PayloadsAllTheThings, MITRE ATT&CK, OWASP WSTG/API et les notes, writeups et rapports personnels autorisés.
- Indexer les sources, versions, dates, licences et empreintes ; conserver pour chaque passage son document, sa section et sa provenance. Prévoir une procédure de mise à jour opérateur, séparée de l’exécution d’une campagne.
- Utiliser bge-m3 localement pour les embeddings (représentations vectorielles qui rendent la recherche sémantique possible). Mesurer son empreinte et son temps sur la machine ; ne pas le laisser co-résider avec le modèle principal si cela dépasse le budget mémoire.
- Tester la qualité de recherche sur des questions connues, le filtrage par source et les documents malformés. Les passages récupérés sont des données non fiables : ils ne modifient pas le scope, les permissions ou les règles d’exécution, même s’ils contiennent des instructions adressées au modèle.

**Livrables.** Ingestion, index local, métadonnées de provenance, tests de recherche et politique de mise à jour/attribution. Aucun document ni embedding n’est envoyé à un service distant.

**Sortie M4.** Recherche répétable et traçable ; citations vérifiables dans les explications ; un document injectant des instructions ne peut pas élargir les capacités de l’agent.

### M5 — Orchestration terminal et posture professeur/autonome

**Objectif.** Relier les modèles et les composants déjà protégés dans un flux de travail utilisable en terminal.

**Travail prévu.**

1. Construire une boucle à états : validation du scope → plan de recon → tâche autorisée → collecte → parseurs → mise à jour du graphe → vérification des preuves → rapport. Les transitions et limites sont décidées par le code.
2. Utiliser le modèle orchestrateur pour choisir parmi des outils structurés ; utiliser le spécialiste pour proposer du texte précis. Le spécialiste n’a pas besoin de *tool calling*. Les modèles sont chargés par lots : déchargement explicite avant bascule, contrôle qu’un seul est résident et journal de la bascule.
3. Autoriser l’exécution autonome dans les limites de temps, de débit, de volume et de destination configurées. Le rapport et chaque étape expliquent le but, le résultat, les limites et la commande manuelle équivalente, sans demander une validation à chaque commande de recon/énumération.
4. Garder les instructions de communication (professeur/autonome) dans le prompt système à créer. Les autorisations, le scope, les preuves et les arrêts restent dans le code.
5. Ne pas exposer en v1 de capacité d’exploitation ou de post-exploitation. L’arrêt avant impact doit être une transition de capacité contrôlée par le runtime, jamais une simple consigne au LLM.

**Point de vigilance.** Le pare-feu borne la destination, mais n’empêche pas à lui seul une action à impact sur une cible pourtant autorisée. La conception de M5 doit donc distinguer les capacités de recon/énumération de celles d’exploitation, sans prétendre qu’un filtre de destination suffit. La v1 ne doit pas fournir un shell privilégié ou illimité au cerveau ; si un besoin de commande libre apparaît, le modèle de phase et de contrôle devra être revu avant de l’exposer.

**Livrables.** CLI locale, machine à états, adaptateurs d’outils, gestion des lots de modèles, prompt système professeur/autonome, sorties pédagogiques, reprise sur erreur et journal d’audit local.

**Sortie M5.** Parcours complet sur un labo explicitement autorisé ; zéro demande de permission par commande de recon/énumération ; arrêt vérifié avant toute capacité à impact ; rapport sans trouvaille non prouvée.

### M6 — Banc d’essai, sécurité et critères de livraison

**Objectif.** Vérifier le comportement du système entier et mesurer ses limites avant de l’utiliser sur un engagement réel.

**Jeux d’essai.** Tests unitaires des validateurs et parseurs ; tests d’intégration avec outils simulés ; puis essais de bout en bout uniquement dans un labo possédé ou explicitement autorisé. Le banc inclut des cas normaux, erreurs, entrées hostiles, reprise après arrêt et changements de scope.

**Critères obligatoires.**

- Scope absent, expiré, contradictoire ou invalide : aucune session d’outil, aucun conteneur de mains et aucun paquet vers la cible.
- Destination hors scope ou exclue : refus côté VM, même si le LLM la propose.
- Cerveau sans route directe vers la cible ; pare-feu testé séparément du prompt.
- Fait extrait reproductible avec provenance ; preuve absente ou empreinte invalide : pas de trouvaille confirmée au rapport.
- Demande d’exploitation/post-exploitation : arrêt contrôlé, aucune capacité correspondante appelée.
- Tentative d’instruction malveillante dans un document RAG : aucun changement de scope ou de politique.
- Résultats des parseurs identiques sur fixtures identiques ; les sorties incomplètes et malformées sont signalées, pas inventées.
- Un seul modèle chargé à la fois ; pas d’OOM sur le jeu complet ; mémoire, latence, tokens/s, temps de bascule et consommation CPU/GPU consignés.
- Scénarios de *tool calling*, dont un piège de sur-appel : appels attendus, schémas valides et absence d’appel indu dûment consignés.
- Toutes les actions visibles et journalisées ; pas de furtivité ni de tentative de contourner une détection.

Les mesures de vitesse seront comparées à la machine réelle et aux attentes de l’utilisateur. On ne fixera pas un seuil de tokens/s arbitraire avant M0 ; en revanche, un seul échec d’une porte de sécurité bloque la livraison v1.

**Livrables.** Suite de tests automatisés, scénarios de labo, compte rendu de benchmark, limites connues, procédure d’arrêt d’urgence et checklist d’autorisation avant campagne.

**Sortie M6.** Tous les critères de sécurité passent, résultats reproductibles et limites documentées. Le passage de M6 autorise une v1 terminale en cadre autorisé ; il ne valide pas l’exploitation ni le post-exploitation. L’interface web locale sera un chantier ultérieur, toujours derrière les mêmes contrôles.

## 4. Arborescence cible indicative

Elle sera créée progressivement ; rien de cette arborescence n’existe encore dans le checkout audité.

```text
config/                 paramètres locaux et exemples inactifs de scope
scripts/                diagnostic et scripts de test
models/                 Modelfiles et manifeste des modèles
src/agent_pentest/      scope, état, parseurs, preuves, RAG, orchestration
ops/vm/                 Compose, pare-feu et procédures de la VM dédiée
prompts/                instructions de communication uniquement
tests/                  tests unitaires, intégration et scénarios synthétiques
docs/                   architecture, décisions, plan et résultats
```

Le vrai `scope.yaml` d’une campagne ne doit pas être fourni comme scope actif par défaut ni contenir une cible choisie implicitement par le programme. Les chemins et le packaging pourront être ajustés au moment de créer le squelette ; les contrôles et critères d’acceptation ci-dessus ne changent pas.

## 5. Informations requises pour lancer M0

À confirmer avant de sélectionner définitivement les modèles et de fabriquer la VM :

1. modèle exact du GPU et VRAM réellement disponible ;
2. OS de l’hôte (Windows/WSL2, Linux ou serveur dédié) ;
3. emplacement prévu de la VM et des conteneurs ;
4. sortie JSON de `python scripts/diagnostic.py --json` une fois le script réintroduit ou recréé et exécuté.

Le diagnostic a été lancé dans le bac à sable Arena : Linux x86_64, environ 3,85 Gio de RAM et aucun GPU détecté par les interfaces disponibles. Cette mesure décrit le bac à sable, pas ton ordinateur, et ne permet donc pas de choisir les modèles. Lance `python scripts/diagnostic.py --json` sur ta machine, puis transmets la sortie avec le modèle exact du GPU, l’OS hôte et l’emplacement prévu de la VM.
