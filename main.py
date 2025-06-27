"""
This is where the actual working of the program will happen!
We'll be Querying stuff for testing whether your classes were defined correctly

Whenever something inappropriate happens (like a function returning false in people.py),
raise a Value Error.
"""
from people import * # import everything!

if __name__ == "__main__":  # Equivalent to int main() {} in C++.
    last_input = 99
    while last_input != 0:
        last_input = int(input("Please enter a query number:"))

        if last_input == 1:
            name = input("Name:")
            ID = int(input("ID:"))
            city = input("City:")
            branchcodes = input("Branch(es):")
            # How will you conver this to a list, given that
            # the user will always enter a comma separated list of branch codes?
            # eg>   2,5
            branchcodes = list(map(int, branchcodes.strip().split(',')))
            salary = input("Salary: ")
            salary = int(salary) if salary else None
            position = input("Position (Junior/Senior/Team Lead/Director): ")
            # Create a new Engineer with given details.
            engineer = Engineer(name, age=int(input("Age: ")), ID=ID, city=city,branchcodes=branchcodes, position=position, salary=salary)

            engineer_roster.append(engineer) # Add him to the list! See people.py for definiton
            
        
        elif last_input == 2:
            # Gather input to create a Salesperson
            # Then add them to the roster
            name = input("Name: ")
            ID = int(input("ID: "))
            city = input("City: ")
            branchcodes = input("Branch(es): ")
            branch_list = list(map(int, branchcodes.strip().split(',')))
            salary_input = input("Salary: ")
            salary = int(salary_input) if salary_input else None
            position = input("Position (Rep/Manager/Head): ")
            superior_input = input("Superior ID (leave blank if none): ")
            superior = int(superior_input) if superior_input else None

            salesman = Salesman(name, age=int(input("Age: ")), ID=ID, city=city,
                                branchcodes=branch_list, position=position, superior=superior, salary=salary)
            sales_roster.append(salesman)

        elif last_input == 3:
            ID = int(input("ID: "))
            # Print employee details for the given employee ID that is given. 
            
            found_employee = None
            for employee in engineer_roster + sales_roster:
                if employee.ID == int(ID):
                    found_employee = employee
                    break
            
            if not found_employee: print("No such employee")
            else:
                print(f"Name: {found_employee.name} and Age: {found_employee.age}")
                print(f"City of Work: {found_employee.city}")
                branch_names = [branchmap[b]["name"] for b in found_employee.branches]
                
                ## Write code here to list the branch names to
                ## which the employee reports as a comma separated list
                ## eg> Branches: Goregaon,Fort1
                print("Branches: " + ",".join(branch_names))
                ## ???? what comes here??
                # print(f"Branches: " + ???? )
                
                print(f"Salary: {found_employee.salary}")

        elif last_input == 4:
            #### NO IF ELSE ZONE ######################################################
            # Change branch to new branch or add a new branch depending on class
            # Inheritance should automatically do this. 
            # There should be no IF-ELSE or ternary operators in this zone
            ID = int(input("ID: "))
            new_branch = int(input("Enter new branch code: "))
            found = False
            for emp in engineer_roster + sales_roster:
                if emp.ID == ID:
                    success = emp.migrate_branch(new_branch)
                    print("Success!" if success else "Failed.")
                    found = True
                    break
            if not found:
                print("No such employee")
            #### NO IF ELSE ZONE ENDS #################################################

        elif last_input == 5:
            ID = int(input("Enter Employee ID to promote: "))
            new_position = input("Enter new position: ")
            for emp in engineer_roster + sales_roster:
                if emp.ID == ID:
                    success = emp.promote(new_position)
                    print("Promoted!" if success else "Promotion failed.")
                    break
            else:
                print("No such employee")

            # promote employee to next position

        elif last_input == 6:
            ID = int(input("Enter Employee ID to give increment: "))
            # Increment salary of employee.
            amt = int(input("Increment amount: "))
            for emp in engineer_roster + sales_roster:
                if emp.ID == ID:
                    emp.increment(amt)
                    print("Increment done.")
                    break
            else:
                print("No such employee")
        
        elif last_input == 7:
            ID = int(input("Enter Employee ID to find superior: "))
            # Print superior of the sales employee.
            for s in sales_roster:
                if s.ID == ID:
                    sid, sname = s.find_superior()
                    if sid is None:
                        print("No superior assigned.")
                    else:
                        print(f"Superior: {sname} (ID: {sid})")
                    break
            else:
                print("No such Salesman")
        
        elif last_input == 8:
            ID_E = int(input("Enter Employee ID to add superior: "))
            ID_S = int(input("Enter Employee ID of superior: "))
            # Add superior of a sales employee
            for s in sales_roster:
                if s.ID == ID_E:
                    success = s.add_superior()
                    print("Superior added." if success else "No valid superior found.")
                    break
            else:
                print("No such Salesman")

        else:
            raise ValueError("No such query number defined")

            
            

            


            


        






