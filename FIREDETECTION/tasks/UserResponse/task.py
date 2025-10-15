[sys.path.append(os.path.join(os.getcwd(), folder)) for folder in variables.get("dependent_modules_folders").split(",")]
import proactive_helper as ph
import json
import random

accept_string    = variables.get("accept_string")
reject_string    = variables.get("reject_string")
case_sensitive   = variables.get("case_sensitive") == "true"
default_response = variables.get("default_response")

class UserResponse:
    def __init__(self):
        pass

    def check_user_response(self, user_response: str) -> bool:
        resp = user_response if case_sensitive else user_response.lower()
        accept = accept_string if case_sensitive else accept_string.lower()
        reject = reject_string if case_sensitive else reject_string.lower()

        if resp == accept:
            return True
        if resp == reject:
            return False
        if default_response == "accept":
            return True
        if default_response == "reject":
            return False
        return False

    def process_selected_users(self, selected_users_json: str) -> list:
        """
        Process selected users and simulate their responses.
        Returns users who accepted the alert.
        """
        try:
            selected_users = json.loads(selected_users_json)
            accepted_users = []
            
            for user in selected_users:
                # Simulate user response (80% acceptance rate)
                accepted = random.random() < 0.8
                
                if accepted:
                    user_with_response = user.copy()
                    user_with_response['accepted'] = True
                    accepted_users.append(user_with_response)
            
            return accepted_users
        except Exception as e:
            print(f"Error processing selected users: {e}")
            return []

if __name__ == '__main__':
    # Get selected users from SelectUsers task
    selected_users_json = variables.get("SELECTED_USERS")
    
    if selected_users_json:
        print(f"Received selected users: {selected_users_json}")
        
        manager = UserResponse()
        accepted_users = manager.process_selected_users(selected_users_json)
        
        print(f"Users who accepted: {len(accepted_users)}")
        print(f"Accepted users: {accepted_users}")
        
        resultMap.put("USER_RESPONSE_ACCEPTED", json.dumps(accepted_users))
    else:
        print("No selected users received from SelectUsers task")
        resultMap.put("USER_RESPONSE_ACCEPTED", json.dumps([]))