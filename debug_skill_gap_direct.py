from pathlib import Path
from src.ingest.esco_loader import create_esco_loader
from src.ingest.text_to_esco_mapper import create_text_mapper
from src.reasoner.skill_gap import SkillGapAnalyzer
from difflib import SequenceMatcher

processed_dir = str(Path('data') / 'processed')
print('Processed dir:', processed_dir)

esco = create_esco_loader(processed_dir)
mapper = create_text_mapper(processed_dir)

user_skills_text = ['Python Programming', 'Data Analysis', 'Machine Learning']
user_skill_ids = set()
for s in user_skills_text:
    matches = mapper.map_text_to_skills(s, top_k=2)
    print(f"Skill '{s}' -> matches: {matches}")
    if matches:
        user_skill_ids.add(matches[0].get('esco_id'))
    else:
        # fallback fuzzy over esco skills
        best = (None, 0.0)
        for esco_id, sdata in esco.skills.items():
            title = sdata.get('title','')
            score = SequenceMatcher(None, s.lower(), title.lower()).ratio()
            if score > best[1]:
                best = (esco_id, score)
        if best[0] and best[1] >= 0.6:
            user_skill_ids.add(best[0])
            print(f"Fallback mapped '{s}' -> {esco.skills[best[0]]['title']} ({best[1]:.2f})")

print('Mapped skill IDs:', user_skill_ids)

# pick a sample occupation
occ_ids = list(esco.occupations.keys())
if not occ_ids:
    print('No occupations loaded')
    exit(1)

job_id = occ_ids[0]
print('Testing job:', job_id, esco.occupations[job_id]['title'])

analyzer = SkillGapAnalyzer(esco_loader=esco, config_path=str(Path('configs') / 'system_config.yaml'))
analysis = analyzer.analyze_path(user_skill_ids, [job_id], model_prob=0.8)

print('Feasibility:', analysis.feasibility_score)
print('Combined:', analysis.combined_score)
print('Per-job gaps:')
for jid,gap in analysis.per_job_gaps.items():
    print(jid, 'required', gap.required_skills, 'missing', gap.missing_skills, 'gap_score', gap.gap_score)

print('Done')
