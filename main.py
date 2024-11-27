import streamlit as st
import yaml
import os
from typing import List
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew
import pandas as pd


# Set page config
st.set_page_config(
    page_title="AI Project Planner",
    page_icon="📋",
    layout="wide"
)

# Pydantic models
class TaskEstimate(BaseModel):
    task_name: str = Field(..., description="Name of the task")
    estimated_time_hours: float = Field(..., description="Estimated time to complete the task in hours")
    required_resources: List[str] = Field(..., description="List of resources required to complete the task")

class Milestone(BaseModel):
    milestone_name: str = Field(..., description="Name of the milestone")
    tasks: List[str] = Field(..., description="List of task IDs associated with this milestone")

class ProjectPlan(BaseModel):
    tasks: List[TaskEstimate] = Field(..., description="List of tasks with their estimates")
    milestones: List[Milestone] = Field(..., description="List of project milestones")

# Load YAML configurations
def load_yaml_config():
    agents_config = {
        'project_planning_agent': {
            'role': 'Project Planning Specialist',
            'goal': 'Break down projects into detailed, actionable tasks',
            'backstory': 'Experienced project manager with expertise in task breakdown and project planning'
        },
        'estimation_agent': {
            'role': 'Estimation Expert',
            'goal': 'Provide accurate time and resource estimates for tasks',
            'backstory': 'Senior technical lead with years of experience in project estimation'
        },
        'resource_allocation_agent': {
            'role': 'Resource Manager',
            'goal': 'Optimize resource allocation across project tasks',
            'backstory': 'Resource optimization specialist with expertise in team management'
        }
    }
    
    tasks_config = {
        'task_breakdown': {
            'description': 'Break down the project into specific tasks',
            'expected_output': 'List of well-defined tasks'
        },
        'time_resource_estimation': {
            'description': 'Estimate time and resources needed for each task',
            'expected_output': 'Detailed estimates for each task'
        },
        'resource_allocation': {
            'description': 'Allocate resources efficiently across tasks',
            'expected_output': 'Complete resource allocation plan'
        }
    }
    
    return agents_config, tasks_config

def create_crew(inputs):
    # Load configurations
    agents_config, tasks_config = load_yaml_config()
    
    # Create agents
    project_planning_agent = Agent(
        config=agents_config['project_planning_agent']
    )
    
    estimation_agent = Agent(
        config=agents_config['estimation_agent']
    )
    
    resource_allocation_agent = Agent(
        config=agents_config['resource_allocation_agent']
    )
    
    # Create tasks
    task_breakdown = Task(
        config=tasks_config['task_breakdown'],
        agent=project_planning_agent
    )
    
    time_resource_estimation = Task(
        config=tasks_config['time_resource_estimation'],
        agent=estimation_agent
    )
    
    resource_allocation = Task(
        config=tasks_config['resource_allocation'],
        agent=resource_allocation_agent,
        output_pydantic=ProjectPlan
    )
    
    # Create and return crew
    return Crew(
        agents=[
            project_planning_agent,
            estimation_agent,
            resource_allocation_agent
        ],
        tasks=[
            task_breakdown,
            time_resource_estimation,
            resource_allocation
        ],
        verbose=True
    )

def main():
    st.title("AI Project Planning Assistant")
    st.write("Enter your project details below to generate a comprehensive project plan.")
    
    with st.form("project_input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            project = st.text_input("Project Name", value="Website")
            industry = st.text_input("Industry", value="Technology")
            project_objectives = st.text_area(
                "Project Objectives",
                value="Create a website for a small business"
            )
        
        with col2:
            team_members = st.text_area(
                "Team Members (One per line, include roles)",
                value="""- John Doe (Project Manager)
- Jane Doe (Software Engineer)
- Bob Smith (Designer)
- Alice Johnson (QA Engineer)
- Tom Brown (QA Engineer)"""
            )
            
            project_requirements = st.text_area(
                "Project Requirements (One per line)",
                value="""- Create a responsive design that works well on desktop and mobile devices
- Implement a modern, visually appealing user interface with a clean look
- Develop a user-friendly navigation system with intuitive menu structure
- Include an "About Us" page highlighting the company's history and values
- Design a "Services" page showcasing the business's offerings with descriptions
- Create a "Contact Us" page with a form and integrated map for communication
- Implement a blog section for sharing industry news and company updates
- Ensure fast loading times and optimize for search engines (SEO)
- Integrate social media links and sharing capabilities
- Include a testimonials section to showcase customer feedback and build trust"""
            )
        
        submit_button = st.form_submit_button("Generate Project Plan")
    
    if submit_button:
        with st.spinner("Generating project plan..."):
            try:
                # Prepare inputs
                inputs = {
                    'project_type': project,
                    'project_objectives': project_objectives,
                    'industry': industry,
                    'team_members': team_members,
                    'project_requirements': project_requirements
                }
                
                # Create and run crew
                crew = create_crew(inputs)
                result = crew.kickoff(inputs=inputs)
                
                # Display results
                st.success("Project plan generated successfully!")
                
                # Display tasks
                st.subheader("Tasks and Estimates")
                tasks_df = pd.DataFrame(result.pydantic.dict()['tasks'])
                st.dataframe(tasks_df)
                
                # Display milestones
                st.subheader("Project Milestones")
                milestones_df = pd.DataFrame(result.pydantic.dict()['milestones'])
                st.dataframe(milestones_df)
                
                # Display usage metrics
                st.subheader("Usage Metrics")
                costs = 0.150 * (crew.usage_metrics.prompt_tokens + 
                               crew.usage_metrics.completion_tokens) / 1_000_000
                st.metric("Estimated Cost", f"${costs:.4f}")
                
                metrics_df = pd.DataFrame([crew.usage_metrics.dict()])
                st.dataframe(metrics_df)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()