# Assignment 3: Mini research report

**Deadline: Sunday, November 8, 2026 at 23:59 Europe/Copenhagen (CET).**

Source: [Group Assignment 3 (All Topics), Absalon](https://absalon.ku.dk/courses/91642/assignments/268756), checked October 19, 2026. The exact API deadline is `2026-11-08T22:59:59Z`, or 23:59:59 Copenhagen time. Submission is a PDF upload through Absalon, per group of 3–4 students.

## Official assignment description

The following reproduces the published assignment wording, with HTML formatting converted to Markdown.

In this group assignment, you will prepare a **mini research report** on a subject **related to the topics** covered in the course. Your task is to **formulate and test a hypothesis** or **examine a research question** that is *not* directly addressed in the existing literature, but that connects to one or more of the **curated papers** discussed across the six course topics.

Please remember that this is **group work**. A practical way to begin is for each group member to review several papers of interest from different topics and propose possible research questions, similar to the reproducibility exercise. Once your group has agreed on a **research question and topic**, you can begin developing and testing your ideas collaboratively, but have a backup plan.

You may also choose to base your work on **papers or materials from the final course topic**, which will be presented in the last week's lectures (not included in student presentations). Since this is a *mini-*research report, your research question or hypothesis should be **focused and manageable** within the available time and resources.

Your report should follow the **typical structure of a deep learning research paper**, including: Abstract, Introduction and Hypothesis/Contributions, Background/Related Work (if not covered in the introduction), Methods (theoretical/experimental), Experiments and Results, Conclusion and Discussion.

**Figures and illustrations** should primarily be **your own**. You may include selected figures from existing papers *only* when necessary (e.g., for comparison or reference), provided that they are **explicitly cited** and used appropriately. However, most method sketches, diagrams, and results should be original.

### Format and Submission Requirements

- Use the **ISBI LaTeX template** and follow all formatting guidelines:
- [Author Instructions (ISBI 2027)](https://biomedicalimaging.org/2027/papers/)
- [Download Template (ZIP)](https://biomedicalimaging.org/2025/wp-content/uploads/sites/106/2024/07/ISBI_template-master.zip)
- **Maximum length:** 4 pages (**excluding** references, figures, and tables).
- Do not adjust the font size, line spacing, or page margins.
- **Submission:** PDF file uploaded via *Absalon*.
- The **front page** must include the group number, all group members' names, and the most relevant baseline/curated paper.
- Include a **short contribution statement** describing each member's role (e.g., writing, implementation, experiments, results). Do not use statements such as "**equal contribution**"!
- Briefly specify each group member’s **initial idea or plan** before the group reaches a final agreement on the idea.
- Provide a **link to your GitHub repository** used for the assignment.

### Additional Notes

- Write using **clear, precise, and formal academic language** (similar to that used in research papers or theses).
- Employ **mathematical notation** where appropriate.
- Review the **exam format**, **AI use policy**, and **evaluation criteria** available on the course information page.

## Evaluation criteria

No assignment-specific rubric is attached in Absalon. Canvas lists 100 possible points, but does not provide criterion weights or a point breakdown. The [official course description](https://kurser.ku.dk/course/ndak24003u/) refers exam assessment to the learning outcomes. The following summarizes those outcomes; it is not an additional assignment rubric.

- Knowledge of advanced models, algorithm design and analysis, and deep-learning theory.
- Ability to implement advanced algorithms with current tools, critically read research, apply research to practical and theoretical problems, and compare methods' strengths and limitations.
- Ability to understand advanced techniques, design and optimize models, and organize independent learning.

Assessment covers the three written assignments and group presentations. Every assignment must pass; the overall work determines the final grade on the Danish 7-point scale. No separate weighting for Assignment 3 is published in the checked sources.

## Course-wide attribution and AI rules

The following reproduces the relevant wording from [Course information](https://absalon.ku.dk/courses/91642/pages/course-information), checked October 19, 2026:

> If your solution contains material from sources other than the assignment text, you must cite the source of the material and any changes you have made. This also applies to material from textbooks, Absalon, etc.
>
> [General UCPH rules](https://kunet.ku.dk/studie/datalogi-ba/Sider/emne.aspx?topicid=20fc0507-633c-455d-85ae-eb53a44d4072): It is not permitted to use large language models such as ChatGPT or similar for exams where all aids are allowed. If you use AI in connection with your exam, it is exam cheating, as your assignment will not be the result of your own independent work.
>
> For programming tasks specifically, the use of GitHub Copilot or similar AI-based programming tools is permitted (see the course description under 'Exam')
>
> You can use LLMs as a learning and research tool when you study. You must write all text in your solutions to the assignments completely by yourself
>
> If your solution uses methods or notation that are not used in the course material, you must specify where you have found the method or notation.
>
> If you are in doubt about plagiarism or citation rules, please ask the teachers or TAs.

## Code

- [`code/`](code/): shared synthetic-data generation, NAVI preparation, video-conditioned SAM 3D training and evaluation. See the [active setup and commands](code/README.md).
- Generator source: [`code/src/video3d/data/generation/`](code/src/video3d/data/generation/); existing generation documentation is preserved under [`code/docs/generation/`](code/docs/generation/). Rendered datasets remain outside version control.

## Local template

The existing template is in [`report/ISBI_template-master/`](report/ISBI_template-master/), with the starting source [`Template_ISBI_latex.tex`](report/ISBI_template-master/Template_ISBI_latex.tex).
