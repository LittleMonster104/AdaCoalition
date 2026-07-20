"""Resume dataset loader for talent acquisition domain"""
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict
import re


class ResumeDataset:
    """
    Loader for resume/CV dataset
    
    Contains resumes and job descriptions for talent matching.
    """
    
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Path to resume data directory
        """
        self.data_dir = Path(data_dir)
        self.resumes = None
        self.jobs = None
        self.use_synthetic = False
    
    def load(self):
        """Load resume dataset"""
        print("Loading Resume dataset...")
        
        # Try to load real Resume Corpus data (*.txt + *.lab files)
        txt_files = list(self.data_dir.glob("*.txt"))
        
        if len(txt_files) > 0:
            print(f"Found {len(txt_files)} resume files, loading real data...")
            self.resumes, self.jobs = self._load_real_data(txt_files)
            print(f"✓ Loaded {len(self.resumes)} real resumes and generated {len(self.jobs)} job postings")
        else:
            print("Warning: Resume data not found. Using synthetic data.")
            self.use_synthetic = True
            self.resumes, self.jobs = self._generate_synthetic_data()
        
        return self
    
    def _load_real_data(self, txt_files: List[Path], max_resumes: int = 500) -> tuple:
        """Load real Resume Corpus data"""
        resumes = []
        
        # Limit to first N files for faster loading
        txt_files = txt_files[:max_resumes]
        
        for txt_file in txt_files:
            resume_id = txt_file.stem
            lab_file = txt_file.parent / f"{resume_id}.lab"
            
            try:
                # Load resume text (handle encoding issues)
                with open(txt_file, 'rb') as f:
                    content = f.read()
                    try:
                        text = content.decode('utf-8')
                    except UnicodeDecodeError:
                        text = content.decode('latin-1', errors='ignore')
                
                # Load job category label
                category = "Unknown"
                if lab_file.exists():
                    with open(lab_file, 'r', encoding='utf-8', errors='ignore') as f:
                        category = f.read().strip()
                
                # Extract skills from text
                skills = self._extract_skills(text)
                
                # Extract experience years
                experience_years = self._extract_experience(text)
                
                # Extract education
                education = self._extract_education(text)
                
                resume = {
                    'resume_id': resume_id,
                    'name': f"Candidate {resume_id}",
                    'text': text[:1000],  # First 1000 chars
                    'category': category,
                    'skills': skills,
                    'experience_years': experience_years,
                    'education': education,
                    'industry': self._category_to_industry(category)
                }
                
                resumes.append(resume)
                
            except Exception as e:
                print(f"Warning: Failed to load {txt_file}: {e}")
                continue
        
        # Generate job postings based on resume categories
        jobs = self._generate_jobs_from_resumes(resumes)
        
        return resumes, jobs
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text"""
        text_lower = text.lower()
        
        skill_keywords = {
            'Python', 'Java', 'JavaScript', 'C++', 'C#', 'SQL', 'HTML', 'CSS',
            'React', 'Angular', 'Node.js', 'Django', 'Flask',
            'Machine Learning', 'Deep Learning', 'Data Analysis', 'Data Science',
            'AWS', 'Azure', 'Docker', 'Kubernetes',
            'Project Management', 'Agile', 'Scrum',
            'Linux', 'Windows', 'Unix',
            'Git', 'Database', 'Oracle', 'MySQL', 'MongoDB',
            'Communication', 'Leadership', 'Problem Solving'
        }
        
        found_skills = []
        for skill in skill_keywords:
            if skill.lower() in text_lower:
                found_skills.append(skill)
        
        return found_skills[:10]  # Limit to 10 skills
    
    def _extract_experience(self, text: str) -> int:
        """Extract years of experience from resume"""
        # Look for patterns like "5 years", "10+ years", etc.
        patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
            r'experience[:\s]+(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s+in'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return min(int(match.group(1)), 30)  # Cap at 30 years
        
        # Default heuristic based on text length and keywords
        if 'senior' in text.lower() or 'lead' in text.lower():
            return 8
        elif 'junior' in text.lower() or 'entry' in text.lower():
            return 2
        else:
            return 5
    
    def _extract_education(self, text: str) -> str:
        """Extract education level from resume"""
        text_lower = text.lower()
        
        if 'phd' in text_lower or 'ph.d' in text_lower or 'doctorate' in text_lower:
            return 'PhD'
        elif 'master' in text_lower or 'mba' in text_lower or 'm.s.' in text_lower:
            return 'Master'
        elif 'bachelor' in text_lower or 'b.s.' in text_lower or 'b.a.' in text_lower:
            return 'Bachelor'
        else:
            return 'Unknown'
    
    def _category_to_industry(self, category: str) -> str:
        """Map job category to industry"""
        category_lower = category.lower()
        
        if 'developer' in category_lower or 'engineer' in category_lower:
            return 'Technology'
        elif 'database' in category_lower or 'administrator' in category_lower:
            return 'Technology'
        elif 'manager' in category_lower or 'project' in category_lower:
            return 'Management'
        elif 'analyst' in category_lower or 'security' in category_lower:
            return 'Technology'
        else:
            return 'General'
    
    def _generate_jobs_from_resumes(self, resumes: List[Dict], num_jobs: int = 50) -> List[Dict]:
        """Generate job postings based on resume data"""
        np.random.seed(44)
        
        # Extract unique categories
        categories = list(set([r['category'] for r in resumes]))
        
        jobs = []
        for i in range(num_jobs):
            # Pick a random category
            category = np.random.choice(categories) if categories else "Unknown"
            
            # Find resumes with this category
            matching_resumes = [r for r in resumes if r['category'] == category]
            
            if matching_resumes:
                # Sample skills from matching resumes
                sample_resume = np.random.choice(matching_resumes)
                required_skills = sample_resume['skills'][:np.random.randint(3, 6)]
                experience_required = max(0, sample_resume['experience_years'] - np.random.randint(0, 3))
                industry = sample_resume['industry']
            else:
                required_skills = []
                experience_required = np.random.randint(2, 8)
                industry = 'Technology'
            
            job = {
                'job_id': i,
                'title': category.replace('_', ' ').title(),
                'category': category,
                'required_skills': required_skills,
                'experience_required': experience_required,
                'industry': industry,
                'description': f"Seeking a {category.replace('_', ' ')} with {experience_required}+ years experience"
            }
            jobs.append(job)
        
        return jobs
    
    def _generate_synthetic_data(self, 
                                 num_resumes: int = 300, 
                                 num_jobs: int = 50) -> tuple:
        """Generate synthetic resume and job data"""
        np.random.seed(44)
        
        skills_pool = [
            'Python', 'Java', 'JavaScript', 'SQL', 'Machine Learning',
            'Data Analysis', 'Project Management', 'Communication',
            'Leadership', 'Problem Solving'
        ]
        
        industries = ['Technology', 'Finance', 'Healthcare', 'Education', 'Retail']
        
        resumes = []
        for i in range(num_resumes):
            resumes.append({
                'resume_id': i,
                'name': f"Candidate {i}",
                'skills': list(np.random.choice(skills_pool, size=np.random.randint(3, 8), replace=False)),
                'experience_years': int(np.random.randint(0, 15)),
                'education': np.random.choice(['Bachelor', 'Master', 'PhD']),
                'industry': np.random.choice(industries)
            })
        
        jobs = []
        for i in range(num_jobs):
            jobs.append({
                'job_id': i,
                'title': f"Position {i}",
                'required_skills': list(np.random.choice(skills_pool, size=np.random.randint(3, 6), replace=False)),
                'experience_required': int(np.random.randint(0, 10)),
                'industry': np.random.choice(industries),
                'description': f"Job description for position {i}"
            })
        
        return resumes, jobs
    
    def get_resume(self, resume_id: int) -> Dict:
        """Get resume details"""
        if isinstance(self.resumes, list):
            return self.resumes[resume_id]
        return self.resumes.get(str(resume_id), {})
    
    def get_job(self, job_id: int) -> Dict:
        """Get job details"""
        if isinstance(self.jobs, list):
            return self.jobs[job_id]
        return self.jobs.get(str(job_id), {})
    
    def get_task_batch(self, batch_size: int = 32) -> List[Dict]:
        """
        Get a batch of talent matching tasks
        
        Args:
            batch_size: Number of tasks
            
        Returns:
            List of matching task dictionaries
        """
        tasks = []
        
        for i in range(min(batch_size, len(self.jobs) if isinstance(self.jobs, list) else len(self.jobs))):
            job = self.get_job(i)
            
            # Sample some candidate resumes
            num_candidates = min(5, len(self.resumes) if isinstance(self.resumes, list) else len(self.resumes))
            candidate_ids = np.random.choice(
                len(self.resumes) if isinstance(self.resumes, list) else len(self.resumes),
                size=num_candidates,
                replace=False
            )
            
            candidates = [self.get_resume(int(cid)) for cid in candidate_ids]
            
            task = {
                'domain': 'talent',
                'task_type': 'candidate_matching',
                'job_id': job.get('job_id', i),
                'job_title': job.get('title', ''),
                'required_skills': job.get('required_skills', []),
                'candidates': candidates,
                'query': f"Rank candidates for job: {job.get('title', '')}"
            }
            tasks.append(task)
        
        return tasks
    
    def get_requirement_vector(self) -> np.ndarray:
        """
        Get typical requirement vector for talent tasks
        
        Returns:
            5-dimensional capability requirement
        """
        # Talent domain emphasizes: reasoning, knowledge, analysis
        return np.array([0.7, 0.8, 0.9, 0.5, 0.6])
