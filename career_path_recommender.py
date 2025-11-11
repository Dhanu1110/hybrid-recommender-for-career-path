#!/usr/bin/env python3
"""CLI wrapper for the Career Path Recommender

Usage examples:
  python career_path_recommender.py --current "Software Engineer" --target "Tech Lead" --skills "Python, System Design" --k 5 --seed 42 --debug

This script loads model artifacts (from artifacts.json if present), initializes the recommender,
and prints top-K recommendations. Explanations via Gemini are optional and controlled with --use_gemini.
"""
import argparse
import logging
import json
import os
from pathlib import Path
from typing import List
import requests

# Ensure src on path
import sys
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.models.career_path_recommender import create_career_recommender


def load_artifacts(models_dir: str):
    artifacts_file = Path(models_dir) / "artifacts.json"
    if artifacts_file.exists():
        try:
            with open(artifacts_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def get_gemini_explanation(api_key: str, prompt: str) -> str:
    """Call Gemini API to get a natural language explanation."""
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    params = {"key": api_key}
    try:
        response = requests.post(url, headers=headers, params=params, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"[AI explanation unavailable: {e}]"

def main():
    parser = argparse.ArgumentParser(description="Career path recommender CLI")
    parser.add_argument('--current', required=True, help='Current job title')
    parser.add_argument('--target', default=None, help='Target job title (optional)')
    parser.add_argument('--skills', default=None, help='Comma-separated skills')
    parser.add_argument('--k', type=int, default=5, help='Top-K recommendations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for inference tie-breaks')
    parser.add_argument('--models_dir', default='models/bert4rec', help='Model directory')
    parser.add_argument('--esco_dir', default='data', help='ESCO data directory')
    parser.add_argument('--use_gemini', action='store_true', help='Append explanations via Gemini (not used for ranking)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--exclude_seen', action='store_true', help='Exclude seen jobs from candidates (default True)')

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger('career_path_recommender_cli')

    # Read artifacts
    artifacts = load_artifacts(args.models_dir)
    checkpoint = artifacts.get('checkpoint_path') or (Path(args.models_dir) / 'checkpoints' / 'bert4rec_tiny.pt')
    vocab = artifacts.get('vocab_path') or (Path(args.models_dir) / 'vocab.json')
    model_config = Path(args.models_dir) / 'model_config.json'

    # Validate gemini key
    gemini_key = os.environ.get('GEMINI_API_KEY')
    if args.use_gemini and not gemini_key:
        logger.warning('--use_gemini passed but GEMINI_API_KEY not set; explanations will be skipped')
        args.use_gemini = False

    # Build user history
    history = [args.current]
    if args.target:
        history.append(args.target)

    skills = []
    if args.skills:
        skills = [s.strip() for s in args.skills.split(',') if s.strip()]

    logger.debug(f"User history tokens: {history}")

    # Create recommender (paths can be absolute or relative)
    recommender = create_career_recommender(model_dir=args.models_dir, esco_data_dir=args.esco_dir)

    mapped_ids = recommender.map_user_jobs_to_model(history)
    logger.debug(f"Mapped token ids before inference: {mapped_ids}")

    # Generate recommendations
    recs = recommender.generate_recommendations(history, user_skills=skills, top_k=args.k, exclude_seen=args.exclude_seen)

    # Optionally append AI explanation (never used for ranking)
    if args.use_gemini:
        for r in recs:
            prompt = f"Explain why the career path from {args.current} to {r['esco_info']['esco_title']} is a good recommendation. Consider the user's skills: {', '.join(skills) if skills else 'N/A'}."
            r['explanation'] = get_gemini_explanation(gemini_key or "AIzaSyA5wlBDBj28RVncJpb6FfhsrdIlHzVRn38", prompt)

    # Print results
    for rec in recs:
        print(f"{rec['rank']}. {rec['model_job_key']} (prob={rec['probability']:.4f}) - ESCO: {rec['esco_info']['esco_title']}")
        if args.use_gemini and rec.get('explanation'):
            print(f"    Explanation: {rec['explanation']}")


if __name__ == '__main__':
    main()
