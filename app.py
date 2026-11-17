
import streamlit as st
import uuid
from dotenv import load_dotenv

load_dotenv()

#### Import all the packages
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from typing import Literal
from send_email import send_mail


st.set_page_config(
    page_title="AI Email Assistant",
    page_icon="📧",
    layout="centered"
)


# --------------------------------------------------
# LLM OBJECT
# --------------------------------------------------

llm = ChatGroq(model="openai/gpt-oss-20b")


# --------------------------------------------------
# GLOBAL STATE
# --------------------------------------------------

class EmailState(BaseModel):
    question: str = ""
    mail_reason: str = ""
    recipient_name: str = ""
    recipient_email: str = ""
    subject: str = ""
    body: str = ""
    feedback: str = ""
    feedback_count: int = 0
    response: str = ""


# --------------------------------------------------
# RETRIEVER NODE
# --------------------------------------------------

class UserDetails(BaseModel):
    mail_reason: str = Field(
        description="Reason of the email"
    )
    recipient_name: str = Field(
        description="Recipient name, if available, else empty string"
    )
    recipient_email: str = Field(
        description="Recipient email address"
    )


def retriever_node(state: EmailState) -> EmailState:

    llm_for_user_details = llm.with_structured_output(UserDetails)

    userDetails: UserDetails = llm_for_user_details.invoke(
        f"Retrieve the user details from this query: {state.question}."
    )

    if not userDetails.mail_reason or not userDetails.recipient_email:
        pass

    state.recipient_name = userDetails.recipient_name
    state.recipient_email = userDetails.recipient_email
    state.mail_reason = userDetails.mail_reason

    return state


# --------------------------------------------------
# DRAFT NODE
# --------------------------------------------------

class DraftEmail(BaseModel):
    subject: str = Field(
        description="Email subject within 40 words"
    )
    body: str = Field(
        description="Email body including proper details, reason and greetings"
    )


def draft_node(state: EmailState) -> EmailState:

    draft_email_llm = llm.with_structured_output(DraftEmail)

    prompt = f"""
    Write an email with these details:

    To: {state.recipient_name}
    Request: {state.mail_reason}

    Please write a proper email body and subject.
    Without extra text.
    Mail body maximum size will be 200 words.
    """

    if state.feedback:

        prompt = f"""
        Revise this email based on the feedback below.

        Current Email:
        Subject: {state.subject}
        Body: {state.body}

        Feedback: {state.feedback}

        Please write a proper email body and subject.
        Improve based on the feedback.
        Mail body maximum size will be 200 words.
        """

    draftEmail: DraftEmail = draft_email_llm.invoke(prompt)

    state.subject = draftEmail.subject
    state.body = draftEmail.body

    return state


# --------------------------------------------------
# REVIEW NODE - HUMAN IN THE LOOP
# --------------------------------------------------

def review_node(state: EmailState) -> EmailState:

    response = interrupt({
        "message": "Do you want to approve or rewrite this email?"
    })

    if response == "yes":
        state.feedback = ""
    else:
        state.feedback = response
        state.feedback_count += 1

    return state


# --------------------------------------------------
# ROUTER
# --------------------------------------------------

def router(state: EmailState) -> Literal["draft", "send", "cancel"]:

    if not state.feedback or state.feedback.strip() == "":
        return "send"

    if state.feedback_count > 2:
        return "cancel"

    return "draft"


# --------------------------------------------------
# CANCEL NODE
# --------------------------------------------------

def cancel_node(state: EmailState) -> EmailState:

    state.response = (
        "Email Not Sent, Because you have reached the "
        "limit of feedbacks"
    )

    return state


# --------------------------------------------------
# SEND NODE
# --------------------------------------------------

def send_node(state: EmailState) -> EmailState:

    res = send_mail(
        state.recipient_email,
        state.subject,
        state.body
    )

    state.response = res

    return state


# --------------------------------------------------
# LANGGRAPH WORKFLOW
# --------------------------------------------------

@st.cache_resource
def create_graph():

    graph = StateGraph(EmailState)

    graph.add_node("retriever", retriever_node)
    graph.add_node("draft", draft_node)
    graph.add_node("send", send_node)
    graph.add_node("cancel", cancel_node)
    graph.add_node("review_node", review_node)

    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "draft")
    graph.add_edge("draft", "review_node")

    graph.add_conditional_edges(
        "review_node",
        router
    )

    graph.add_edge("send", END)
    graph.add_edge("cancel", END)

    return graph.compile(
        checkpointer=InMemorySaver()
    )


final_graph = create_graph()


# --------------------------------------------------
# STREAMLIT SESSION STATE
# --------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "active" not in st.session_state:
    st.session_state.active = False


config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}


# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------

st.title("📧 AI Email Assistant")

st.markdown(
    "### Generate, review, improve and send emails using AI."
)

st.caption(
    "Powered by Groq + LangGraph + Streamlit"
)

st.divider()


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

st.subheader("✍️ Compose Your Email")

with st.form("email_form"):

    query = st.text_area(
        "What email do you want to send?",
        placeholder=(
            "Example: Write an email to Rahul "
            "(rahul@gmail.com) requesting leave "
            "for two days due to personal reasons."
        ),
        height=130
    )

    generate = st.form_submit_button(
        "Generate Email ✨",
        use_container_width=True
    )


if generate:

    if not query.strip():
        st.warning("Please enter your email request.")

    else:

        # Create a fresh conversation for every email
        st.session_state.thread_id = str(uuid.uuid4())

        config = {
            "configurable": {
                "thread_id": st.session_state.thread_id
            }
        }

        with st.spinner("Generating your email..."):

            final_graph.invoke(
                {"question": query},
                config=config
            )

        st.session_state.active = True
        st.rerun()


# --------------------------------------------------
# REVIEW GENERATED EMAIL
# --------------------------------------------------

if st.session_state.active:

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }

    state = final_graph.get_state(config)

    values = state.values

    st.divider()

    # Check whether graph is waiting for human approval
    if state.next:

        st.subheader("📩 Review Your Email")

        st.write("**Recipient:**", values["recipient_name"])
        st.write("**Email Address:**", values["recipient_email"])

        st.text_input(
            "Subject",
            value=values["subject"],
            disabled=True
        )

        st.text_area(
            "Email Body",
            value=values["body"],
            height=250,
            disabled=True
        )

        st.info(
            f"Revision attempts: {values['feedback_count']} / 3"
        )

        st.markdown("### Human Approval")

        with st.form(
            f"review_form_{st.session_state.thread_id}_{values['feedback_count']}"
        ):

            feedback = st.text_area(
                "Provide feedback to improve the email",
                placeholder=(
                    "Example: Make it more professional "
                    "and keep it shorter."
                )
            )

            col1, col2 = st.columns(2)

            with col1:
                approve = st.form_submit_button(
                    "✅ Approve & Send",
                    use_container_width=True
                )

            with col2:
                revise = st.form_submit_button(
                    "🔄 Revise Email",
                    use_container_width=True
                )

        if approve:

            with st.spinner("Sending email..."):

                final_graph.invoke(
                    Command(resume="yes"),
                    config=config
                )

            st.rerun()

        if revise:

            if not feedback.strip():
                st.warning("Please provide feedback.")

            else:

                with st.spinner("Revising your email..."):

                    final_graph.invoke(
                        Command(resume=feedback),
                        config=config
                    )

                st.rerun()

    else:

        # Workflow completed
        response = values.get("response", "")

        if "successfully" in response.lower():

            st.success("🎉 " + response)

            st.balloons()

        else:

            st.error(response or "Email workflow completed.")

        if st.button("➕ Create Another Email"):

            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.active = False
            st.rerun()


st.divider()

st.caption("AI Email Assistant | Built with LangGraph and Streamlit")
