# RAG Evaluation Report

- Questions with answer metrics: **58**
- Answer metrics: `results.json`
- Retrieval metrics: `retrieval_results.json`

## Overall metric summary

| Metric | Mean score | Scored / total |
|---|---:|---:|
| `faithfulness` | 0.895 | 58 / 58 |
| `answer_relevancy` | 0.903 | 58 / 58 |
| `contextual_recall` | 0.928 | 58 / 58 |
| `answer_correctness` | 0.883 | 58 / 58 |
| `citation_correctness` | 0.966 | 58 / 58 |
| `retrieval_recall_at_k` | 0.575 | 58 / 58 |

## Mean scores grouped by family

| Group | Questions | faithfulness | answer_relevancy | contextual_recall | answer_correctness | citation_correctness | retrieval_recall_at_k |
|---|---:|---:|---:|---:|---:|---:|---:|
| cross | 21 | 0.904 | 0.871 | 0.937 | 0.900 | 0.952 | 0.667 |
| single | 22 | 0.924 | 0.928 | 0.970 | 0.886 | 0.955 | 0.818 |
| summary | 15 | 0.840 | 0.909 | 0.853 | 0.853 | 1.000 | 0.089 |

## Mean scores grouped by provider

| Group | Questions | faithfulness | answer_relevancy | contextual_recall | answer_correctness | citation_correctness | retrieval_recall_at_k |
|---|---:|---:|---:|---:|---:|---:|---:|
| groq | 58 | 0.895 | 0.903 | 0.928 | 0.883 | 0.966 | 0.575 |

## Scores by question

| ID | Family | Provider | Top-k | Question | faithfulness | answer_relevancy | contextual_recall | answer_correctness | citation_correctness | retrieval_recall_at_k |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | single | groq | 3 | How does the Supreme Court's ruling on citizenship revocation under Section 8(2) affect statutory residency requirements, and what are the key implications for individuals under this provision? | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | single | groq | 3 | Does the judgment permit Section 5(1)(f) applications with the twelve-month residence requirement under Central Government relaxation? | 0.667 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 3 | single | groq | 3 | How might a Section 5 application be affected if adverse police/security material contradicts the statutory setting? | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4 | single | groq | 3 | How might the case's interaction with Section 5 applications be altered by adverse police/security material? | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 5 | single | groq | 3 | How did the High Court rule on the petitioner's citizenship acquisition under Section 4, considering it could not be treated as a voluntary act when she was a minor? | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 6 | single | groq | 3 | How did the High Court determine that the petitioner's acquisition of Singapore citizenship as a minor was not a voluntary act? | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 7 | single | groq | 3 | What is the URL of the Supreme Court judgment for the case identified in the primary source [21809_2023_6_1502_56499_Judgement_18-Oct-2024.pdf]? | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8 | single | groq | 3 | How do passport surrender procedures differ from citizenship acquisition under Indian legal frameworks? | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 9 | single | groq | 3 | If a person gains foreign citizenship, should separate records be kept for citizenship and passport surrender formalities? | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 10 | single | groq | 3 | How does the Indian citizenship application process determine if someone is already a citizen? | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 11 | single | groq | 3 | How do Indian citizenship rules require specific evidence collection to establish eligibility through statutory pathways? | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| 12 | single | groq | 3 | How does Section 7D's requirement for a hearing opportunity interact with foreigner-control laws in OCI cancellation? | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 13 | single | groq | 3 | How does Section 7D function within OCI cancellation procedures under foreigner-control laws and what hearing rights are associated with this section? | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 14 | single | groq | 3 | Which companion document provides precedent for birth, descent, registration, foreign citizenship, OCI, administrative delays, and deprivation in Indian citizenship cases? | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 15 | single | groq | 3 | How should the Indian Citizenship Case Law: Applicant-Focused Precedent Handbook be cross-referenced for addressing administrative delays and deprivation issues? | 1.000 | 1.000 | 1.000 | 0.300 | 1.000 | 0.000 |
| 16 | single | groq | 3 | What specific residence evidence is required for naturalisation applications under the document matrix rules? | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 17 | single | groq | 3 | How does the Indian Citizenship Act define descent-based citizenship qualification, including the specific conditions under which an individual can claim citizenship through an Indian parent born abroad? | 1.000 | 0.000 | 1.000 | 0.300 | 0.000 | 0.000 |
| 18 | single | groq | 3 | How do Section 6 and the Third Schedule of the Citizenship Act address naturalisation requirements for applicants with Indian parentage? | 1.000 | 0.500 | 1.000 | 0.300 | 1.000 | 0.000 |
| 19 | single | groq | 3 | How many specific processing stages exist for a naturalisation application, including submission and acknowledgement, initial scrutiny for completeness, document verification, police/security inquiry, clarification requests, statutory eligibility assessment, oath of allegiance, and final grant or rejection? | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 20 | single | groq | 3 | How does document verification progress to the next stage in naturalisation processing? | 0.667 | 1.000 | 1.000 | 0.600 | 1.000 | 1.000 |
| 21 | single | groq | 3 | How does Section 7 of the citizenship acquisition rules apply to territories joining India, and what are the specific order specifications and effective dates for this process? | 0.889 | 0.917 | 0.333 | 1.000 | 1.000 | 1.000 |
| 22 | single | groq | 3 | If a territory joins India under Section 7, how would the Central Government's specified order determine citizenship eligibility and effective dates? | 0.714 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 23 | cross | groq | 3 | How does Section ... differ from standard Section ... applications under the Citizenship Act, 1955? | 1.000 | 0.000 | 1.000 | 0.300 | 0.000 | 0.000 |
| 24 | cross | groq | 3 | How does Section 6A's distinct framework interact with constitutional validity and the Assam Accord? | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 25 | cross | groq | 3 | How does Section 7D's OCI cancellation process interact with foreigner-law actions under the Indian legal framework? | 1.000 | 0.917 | 1.000 | 1.000 | 1.000 | 1.000 |
| 26 | cross | groq | 3 | How does Section 7D's reasonable hearing requirement interact with foreigner-law actions for OCI cardholders, and what specific legal provisions might apply to this situation? | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 |
| 27 | cross | groq | 3 | How does the Supreme Court's decision differentiate OCI holders' admission policies from Indian citizens' rights? | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 |
| 28 | cross | groq | 3 | Does OCI status confer full citizenship rights under Indian law, and how does it differ from citizenship in terms of educational admissions? | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 29 | cross | groq | 3 | How does marriage to an Indian citizen compare to the requirements for Section .5(1)(f) citizenship eligibility, and what additional conditions must be satisfied? | 0.867 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 30 | cross | groq | 3 | How do the Section 5(1)(f) requirements for Indian citizenship compare to the conditions for marriage to an Indian citizen? | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 31 | cross | groq | 3 | How do the case limits and residence requirements differ for children born abroad in citizenship registration for different countries? | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 32 | cross | groq | 3 | How do the specific residence rules in Section 5 categories affect citizenship applications, and what are the key differences between these rules and those in other sections? | 0.857 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 |
| 33 | cross | groq | 3 | How do passport authorities handle citizenship disputes for individuals born in India under Section 3? How can I find out more about the legal process for resolving such disputes without relying on a single document? | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 34 | cross | groq | 3 | Do Indian citizens born under Section 12 of the Citizenship Act, 1958 face passport disputes without citizenship verification? How can this be resolved using both the Citizenship Act and the Passport Act? | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 |
| 35 | cross | groq | 3 | How does Section 6B's Collector-centered process differ from older citizenship categories in procedural requirements? | 0.900 | 1.000 | 0.333 | 1.000 | 1.000 | 0.000 |
| 36 | cross | groq | 3 | How did the Supreme Court case Union of India v. Pranav Srinivasan cite specific provisions of the Citizenship Act, 1955? How can the case be interpreted without relying on a single document? | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 37 | cross | groq | 3 | How do the Citizenship Rules, 2009 and the Supreme Court case Union of India v. Pranav Srinivasan relate to each other? | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 |
| 38 | cross | groq | 3 | Which handbook section addresses citizenship by descent and foreign citizenship cases, and how does this compare to the requirements for naturalization in the other handbook? | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 |
| 39 | cross | groq | 3 | How does the Supreme Court's Izhar Ahmad Khan ruling clarify Section 9's voluntary citizenship acquisition termination process, and what specific legal provisions from other relevant documents support this interpretation? | 0.857 | 0.857 | 1.000 | 1.000 | 1.000 | 0.500 |
| 40 | cross | groq | 3 | How do Section 6A's generic citizenship pathways differ from Section 6B's specialized migration categories under CAA? | 0.909 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| 41 | cross | groq | 3 | How do the specified countries, communities, cut-off date, and modified naturalisation period meet Section 6B's eligibility criteria for CAA citizenship? | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 42 | cross | groq | 3 | How do the key areas covered in the Indian Citizenship Case Law guide for applicants differ from the eligibility and application processes outlined in the document? | 0.750 | 0.667 | 1.000 | 0.600 | 1.000 | 0.000 |
| 43 | cross | groq | 3 | How does the Applicant-Focused Precedent Handbook ensure applicants verify current official text before legal decisions, and what specific steps should they take to avoid outdated references in their applications? | 0.444 | 1.000 | 0.333 | 1.000 | 1.000 | 0.000 |
| 44 | summary | groq | 3 | How can I get Indian citizenship through birth, descent, or registration under the Citizenship Act? | 1.000 | 1.000 | 1.000 | 0.900 | 1.000 | 0.000 |
| 45 | summary | groq | 3 | How can I analyze judicial rulings on citizenship delays and deprivation under Section 5(1)(f) and Section 10 to get a high-level overview of the main themes? | 1.000 | 0.444 | 1.000 | 0.300 | 1.000 | 0.000 |
| 46 | summary | groq | 3 | How do Part I through X of the Indian citizenship case-law handbook differ in their focus and purpose? | 0.250 | 0.250 | 0.667 | 0.800 | 1.000 | 0.000 |
| 47 | summary | groq | 3 | How does the handbook categorize citizenship by birth versus citizenship by descent cases? | 0.000 | 1.000 | 1.000 | 0.600 | 1.000 | 0.167 |
| 48 | summary | groq | 3 | How does the citizenship case-law handbook categorize cases by citizenship acquisition and disputes? | 1.000 | 1.000 | 1.000 | 0.800 | 1.000 | 0.167 |
| 49 | summary | groq | 3 | How can I get a high-level overview of the Citizenship Act's eligibility provisions, including specific exceptions for illegal migrants under Section :6B, and how do categories like Indian origin, marriage to Indian citizens, and territorial integration events define citizenship registration? | 0.857 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| 50 | summary | groq | 3 | How can I identify the specific territorial event and notification for citizenship by territorial integration applications? | 1.000 | 1.000 | 0.333 | 1.000 | 1.000 | 0.167 |
| 51 | summary | groq | 3 | Summarize the key historical law amendments that have impacted citizenship by birth determination for individuals born decades ago. | 0.857 | 1.000 | 0.667 | 1.000 | 1.000 | 0.167 |
| 52 | summary | groq | 3 | How can specific territorial integration events and statutory conditions under Section 5 of the Citizenship Act determine eligibility for citizenship registration, including Indian origin, marriage to Indian citizens, minors, and other specified relationships? | 1.000 | 1.000 | 0.800 | 1.000 | 1.000 | 0.167 |
| 53 | summary | groq | 3 | How can I analyze the territorial integration event and specific notification for citizenship claims based on historical residence to get a high-level overview? | 1.000 | 1.000 | 0.667 | 0.900 | 1.000 | 0.167 |
| 54 | summary | groq | 3 | How can I get a high-level overview of Indian citizenship eligibility criteria and application processes across different categories, and what are the key processing stages involved in naturalisation and registration? | 0.760 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| 55 | summary | groq | 3 | What are the seven key procedural stages in Indian citizenship registration as per case law? | 0.875 | 1.000 | 0.667 | 0.800 | 1.000 | 0.000 |
| 56 | summary | groq | 3 | How do the Indian Citizenship Act registration categories differ, and what eligibility criteria apply to each category? | 1.000 | 0.944 | 1.000 | 0.800 | 1.000 | 0.167 |
| 57 | summary | groq | 3 | How can delays in Indian citizenship applications be addressed through verification, documentation, and legal remedies? | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| 58 | summary | groq | 3 | How does the handbook categorize citizenship routes through its structured sections on birth, descent, registration, naturalisation, OCI, and administrative procedures? | 1.000 | 1.000 | 1.000 | 0.900 | 1.000 | 0.167 |
