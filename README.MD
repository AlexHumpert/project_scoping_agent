# AI Project Planner

An intelligent project planning application that leverages AI agents to break down projects, estimate tasks, and allocate resources effectively. Built with Streamlit and CrewAI.


## 🎯 Problem Statement

This application addresses several critical challenges in project planning and management:

1. **Time-Consuming Initial Planning**
   - Traditional project planning requires extensive manual effort to break down projects into tasks
   - Project managers often spend days or weeks creating initial project estimates
   - Resource allocation typically involves multiple meetings and iterations

2. **Inconsistent Estimation**
   - Different team members may estimate similar tasks differently
   - Historical data isn't consistently leveraged for estimation
   - Cognitive biases often affect human estimation accuracy

3. **Resource Allocation Complexity**
   - Balancing team member workloads is challenging
   - Skills matching to tasks requires detailed knowledge of team capabilities
   - Resource conflicts are often discovered late in the planning process

4. **Project Planning Expertise Gap**
   - Not all teams have access to experienced project managers
   - Junior project managers may miss critical tasks or dependencies
   - Small teams often lack formal project planning processes

This AI Project Planner solves these problems by:
- Automating the initial project breakdown process
- Providing consistent, data-driven task estimates
- Optimizing resource allocation based on team composition
- Offering expert-level project planning guidance
- Reducing planning time from days to minutes
- Creating standardized, repeatable planning processes

## 🌟 Features

- **Project Configuration**
  - Define project name, industry, and objectives
  - Input team member details with roles
  - Specify detailed project requirements

- **AI-Powered Planning**
  - Automated task breakdown
  - Intelligent time estimation
  - Resource allocation optimization
  - Milestone creation and organization

- **Visual Results**
  - Interactive data tables for tasks and milestones
  - Usage metrics and cost estimation
  - Clear progress indicators

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-project-planner.git
cd ai-project-planner
```

2. Install required dependencies:
```bash
pip install streamlit crewai pydantic pandas python-dotenv
```

3. Create a `.env` file in the project root and add your OpenAI API key:
```bash
OPENAI_API_KEY=your-api-key-here
```

### Running the Application

1. Start the Streamlit server:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

## 💡 Usage

1. **Enter Project Details**
   - Project name and industry
   - Project objectives
   - Team member information
   - Project requirements

2. **Generate Plan**
   - Click "Generate Project Plan" to start the AI analysis
   - Wait for the processing to complete

3. **Review Results**
   - Examine the generated tasks and their estimates
   - Review project milestones
   - Check usage metrics and costs

## 🤖 AI Agents

The application uses three specialized AI agents:

1. **Project Planning Specialist**
   - Breaks down projects into detailed, actionable tasks
   - Creates logical task sequences

2. **Estimation Expert**
   - Provides time estimates for tasks
   - Determines required resources

3. **Resource Manager**
   - Optimizes resource allocation
   - Creates efficient project schedules

## 📊 Output Format

The application generates a structured project plan including:

- Task list with time estimates and required resources
- Project milestones with associated tasks
- Resource allocation recommendations
- Usage metrics and cost estimates

## 🔒 Security

- API keys are stored securely in environment variables
- No sensitive project data is stored permanently
- All processing is done in real-time

## ⚠️ Limitations

- Requires active internet connection
- Processing time varies based on project complexity
- Costs are based on OpenAI API usage

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [CrewAI](https://github.com/joaomdmoura/crewAI)
- Uses OpenAI's API for AI processing
