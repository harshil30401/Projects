package ATM;

import java.text.DecimalFormat;
import java.util.Scanner;

// All the methods will be defined in this class 
// we can also create 2 classes one for Current Account Handling and one for Saving Account Handling.

public class Account {

    private int customerNumber;
    private int pinNumber;
    private int currentBalance;
    private int savingBalance;

    Scanner input = new Scanner(System.in);

    DecimalFormat moneyFormat = new DecimalFormat("'$'###,##0.00");

    public int setCustomerNumber(int customerNumber) {
        this.customerNumber = customerNumber;
        return customerNumber;
    }

    public int getCustomerNumber() {
        return customerNumber;
    }

    public int setPinNumber(int pinNumber) {
        this.pinNumber = pinNumber;
        return pinNumber;
    }

    public int getPinNumber() {
        return pinNumber;
    }

    public double getSavingBalance() {
        return savingBalance;
    }

    public double getCurrentBalance() {
        return currentBalance;
    }

    // Current Account
    public double calcCurrentWithdraw(double amount) {

        currentBalance -= amount;
        return currentBalance;

    }

    public double calcCurrentDeposit(double amount) {
        currentBalance += amount;
        return currentBalance;
    }

    // Savings Account
    public double calcSavingWithdraw(double amount) {

        savingBalance -= amount;
        return savingBalance;
    }

    public double calcSavingDeposit(double amount) {

        savingBalance += amount;
        return savingBalance;
    }


    // Returning the calculated amount
    public void getCurrentWithdrawInput(){

        System.out.println("Your Current Balance is: "+currentBalance);

        System.out.println("Enter the Amount that you want to Withdraw: ");
        double amount = input.nextDouble();

        if ( (currentBalance - amount) >= 0 ) {
            calcCurrentWithdraw(amount);
            System.out.println("Your new Current Balance is: " + currentBalance);
        }else{
            System.out.println( "Invalid Amount!" );
        }
        
    }
    public void getCurrentDepositInput(){

        System.out.println("Your Current Balance is: "+currentBalance);

        System.out.println("Enter the Amount that you want to Deposit: ");
        double amount = input.nextDouble();

        if ( (amount) >= 0 ) {
            calcCurrentDeposit(amount);
            System.out.println("Your new Current Balance is: " + currentBalance);
        }else{
            System.out.println( "Invalid Amount!" );
        }

    }

    public void getSavingWithdrawInput(){

        System.out.println("Your Saving Balance is: "+savingBalance);

        System.out.println("Enter the Amount that you want to Withdraw: ");
        double amount = input.nextDouble();

        if ( (savingBalance - amount) >= 0 ) {
            calcSavingWithdraw(amount);
            System.out.println("Your new Saving Balance is: " + savingBalance);
        }else{
            System.out.println( "Invalid Amount!" );
        }
        
    }
    public void getSavingDepositInput(){

        System.out.println("Your Saving Balance is: "+savingBalance);

        System.out.println("Enter the Amount that you want to Withdraw: ");
        double amount = input.nextDouble();

        if ( (currentBalance + amount) >= 0 ) {
            calcSavingDeposit(amount);
            System.out.println("Your new Saving Balance is: " + currentBalance);
        }else{
            System.out.println( "Invalid Amount!" );
        }

    }


}
