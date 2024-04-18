import re

# Defining the function to process the strings
def source_conversation_data(conversation_str):
    """Given a raw conversation source text - build a structured conversation data

    Args:
        - conversation_str (str): conversation source str object from csv

    Returns:
        - conversation_list list(dict): data object with human-ai annotated text
    """
    # Step 1: Preprocess the string to insert a separator before 'human_message:' and 'ai_message:'
    processed_str = re.sub(r"(human_message:|ai_message:)", r"|||\1", conversation_str)
    
    # Step 2: Splitting the string into segments based on our custom separator
    segments = processed_str.split("|||")
    
    # Step 3: Remove the first empty segment if it exists
    if segments[0] == '':
        segments = segments[1:]
    
    # Step 4: Iterate through the segments and classify each as 'human' or 'ai'
    conversation_list = []
    for segment in segments:
        # Step 5: Determine the type of message and extract content
        if 'human_message:' in segment:
            message_type = 'human'
            content = segment.replace('human_message:', '').strip()
        elif 'ai_message:' in segment:
            message_type = 'ai'
            content = segment.replace('ai_message:', '').strip()
        else:
            continue  # Ignore any segment that does not match expected patterns
        
        # Step 6: Append the structured dictionary to the list
        conversation_list.append({message_type: content})
    
    # Step 7: Return the conversation_list
    return conversation_list


def context_conversation_data(conversation_str):
    """Given a raw conversation source text - build a structured conversation data

    Args:
        - conversation_str (str): conversation source str object from csv

    Returns:
        - conversation_list list(dict): data object with human-ai annotated text
    """
    # Step 1: Preprocess the string to insert a separator before 'Customer's Message:' and 'Agent's Message:'
    processed_str = re.sub(r"(Customer's Message:|Agent's Message:)", r"|||\1", conversation_str)
    
    # Step 2: Splitting the string into segments based on our custom separator
    segments = processed_str.split("|||")
    
    # Step 3: Remove the first empty segment if it exists
    if segments[0] == '':
        segments = segments[1:]
    
    # Step 4: Iterate through the segments and classify each as 'human' or 'ai'
    conversation_list = []
    for segment in segments:
        # Step 5: Determine the type of message and extract content
        if "Customer's Message:" in segment:
            message_type = 'customer'
            content = segment.replace("Customer's Message:", '').strip()
        elif "Agent's Message:" in segment:
            message_type = 'agent'
            content = segment.replace("Agent's Message:", '').strip()
        else:
            continue  # Ignore any segment that does not match expected patterns
        
        # Step 6: Append the structured dictionary to the list
        conversation_list.append({message_type: content})
    
    # Step 7: Return the conversation_list
    return conversation_list
