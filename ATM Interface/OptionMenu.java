package ATM;

import java.text.DecimalFormat;
import java.io.IOException;
import java.util.*;

public class OptionMenu extends Account {

    Scanner menuInput = new Scanner(System.in);

    // This is used to denote what kind of money format we can expect
    DecimalFormat moneyFormat = new DecimalFormat("'$'###,##0.00");

    // Using HashMap to store the key : value pairs of the Account Number and Pin
    // Number
    HashMap<Integer, Integer> data = new HashMap<Integer, Integer>();

    // First Requirement: " Should create a new account with Zero Balance "
    public void createAccount() {
        System.out.println("New Account Created!");
    }

    public void getLogin() throws IOException{
        System.out.println("Welcome!");
        int x = 1;
        do {
            try {

                // This is the First Entry, as of now for testing purposes we can add custom
                data.put(1234, 5678);
                
                System.out.println("--------------------------------------------");

                System.out.print("Please enter your ID: ");
                setCustomerNumber(menuInput.nextInt());

                System.out.print("Please enter your PIN: ");
                setPinNumber(menuInput.nextInt());

                System.out.println("--------------------------------------------");

            } catch (Exception e) {
                System.out.println("An Error occured! ");
                System.out.println(e);
                x = 2;
            }

            int cn = getCustomerNumber();
            int pn = getPinNumber();

            if (data.containsKey(cn) && (data.get(cn) == pn)) {
                getAccountType();
            }

        } while (x == 1);
    }

    public void getAccountType() {
        System.out.println("Select the Account that you want to Access: ");
        System.out.println("Type 1 - Current Account");
        System.out.println("Type 2 - Savings Account");
        System.out.println("Type 3 - Exit");

        int selection = menuInput.nextInt();

        switch (selection) {
            case 1:

                getCurrent();
                break;

            case 2:

                getSaving();
                break;

            case 3:
                System.out.println("Thank you for using this ATM!");
                break;

            default:
                System.out.println("Invalid choice!");
                break;
        }

    }

    public void getCurrent() {
        System.out.println("Current Account: ");
        System.out.println("Type 1 - View Balance");
        System.out.println("Type 2 - Withdraw Funds");
        System.out.println("Type 3 - Deposit Funds");
        System.out.println("Type 4 - Exit");

        int selection = menuInput.nextInt();

        switch (selection) {
            case 1:
                System.out.println("Current Account Balance: " + moneyFormat.format(getCurrentBalance()));
                getAccountType();
                break;

            case 2:
                getCurrentWithdrawInput();
                getAccountType();
                break;

            case 3:
                getCurrentDepositInput();
                getAccountType();
                break;
            case 4:
                System.out.println("Thank you for using the ATM!");
                break;

            default:
                System.out.println("Invalid choice!");
                break;
        }

    }

    public void getSaving() {
        System.out.println("Saving Account: ");
        System.out.println("Type 1 - View Balance");
        System.out.println("Type 2 - Withdraw Funds");
        System.out.println("Type 3 - Deposit Funds");
        System.out.println("Type 4 - Exit");

        int selection = menuInput.nextInt();

        switch (selection) {
            case 1:
                System.out.println("Saving Account Balance: " + moneyFormat.format(getSavingBalance()));
                getAccountType();
                break;

            case 2:
                getSavingWithdrawInput();
                getAccountType();
                break;

            case 3:
                getSavingDepositInput();
                getAccountType();
                break;
            case 4:
                System.out.println("Thank you for using the ATM!");
                break;

            default:
                System.out.println("Invalid choice!");
                break;
        }

    }
}
