from pathlib import Path
from src.ingest.text_to_esco_mapper import create_text_mapper
from src.ingest.esco_loader import create_esco_loader
from src.reasoner.skill_gap import create_skill_gap_analyzer

processed_dir = str(Path('data') / 'processed')
print('Processed dir:', processed_dir)

esco = create_esco_loader(processed_dir)
mapper = create_text_mapper(processed_dir)

user_skills_text = ['Python Programming', 'Data Analysis', 'Machine Learning']
mapped_skills = []
for s in user_skills_text:
    matches = mapper.map_text_to_skills(s, top_k=2)
    print(f"Skill '{s}' -> matches: {matches}")
    if matches:
        mapped_skills.append(matches[0]['esco_id'])

print('Mapped skills IDs:', mapped_skills)

analyzer = create_skill_gap_analyzer(str(Path('configs') / 'system_config.yaml'))
# attach loader if factory didn't
analyzer.esco_loader = esco

candidate_paths = [(['occ_001','occ_002'], 0.85)]
user_skill_set = set(mapped_skills)

analyses = analyzer.analyze_multiple_paths(user_skill_set, candidate_paths)
for a in analyses:
    print('Path:', a.path)
    print('Model prob:', a.model_prob)
    print('Feasibility:', a.feasibility_score)
    print('Combined:', a.combined_score)
    for jid,gap in a.per_job_gaps.items():
        print(' Job', jid, 'required:', gap.required_skills, 'missing:', gap.missing_skills, 'gap_score:', gap.gap_score)

print('Done')
