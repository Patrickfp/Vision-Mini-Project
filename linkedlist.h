#include <iostream>

struct node
{
	int value = 0;
	node* next = NULL;
};
class llist
{
public:
	llist()
	{
		head = NULL;
		tail = NULL;
	}
	bool isEmpty() { return head == NULL;}
	void append(int value)
	{
		node *temp = new node;
		temp->value = value;
		temp->next = NULL;
		if (isEmpty())
		{
			head = temp;
			tail = temp;
		}
		else
		{
			tail->next = temp;
			tail = temp;
		}
	}
	void appendList(llist l)
	{
		node* temp = l.head;
		while (temp != NULL)
		{
			insertSort(temp->value);
			temp = temp->next;
		}
	}
	void insertSort(int value)
	{
		node *temp = new node;
		if (isEmpty())
		{
			head = temp;
			tail = temp;
		}
		else if (head->value > value)
		{
			temp->value = value;
			temp->next = head;
			head = temp;
		}
		else if(tail->value < value)
        {
		    temp->value = value;
		    tail->next = temp;
            tail = temp;
        }
		else
		{
			temp = head;
			while (temp->next != NULL && temp->next->value < value)
			{
				temp = temp->next;
			}
			node *temp2 = new node;
			temp2->value = value;
			temp2->next = temp->next;
			temp->next = temp2;
			if (temp2->next == NULL)
			    tail = temp2;
		}
	}
	int median()
	{
		int tempval = 0;
		node *temp = head;
		node *temp2 = head;
		if (isEmpty())
			return 0;
		while (temp2 != NULL && temp2->next != NULL)
		{
            tempval = temp->value;
			temp = temp->next;
			temp2 = temp2->next->next;
		}
        if (temp2 == NULL)
            return (temp->value+tempval)/2;
		return temp->value;
	}
	int average()
	{
		if (isEmpty())
			return 0;
		node *temp = head;
		int sum = 0;
		int ite = 0;
		while (temp != NULL)
		{
			sum += temp->value;
			ite++;
			temp = temp->next;
		}
		//std::cout << sum / (float)ite << std::endl;
		//std::cout << ite << "\n";
		return sum / ite;
	}
	int min()
    {
	    if (head == NULL)
	        return 0;
	    return head->value;

    }
    int max()
    {
	    if (tail == NULL)
	        return 0;
	    return tail->value;
    }
	void display()
	{
		node *temp = new node;
		temp = head;
		while (temp != NULL)
		{
			std::cout << temp->value << "\t";
			temp = temp->next;
		}
	}
	~llist()
    {
	    //std::cout << "hej";
        node *temp = head;
        while(temp != NULL)
        {
            node *temp2 = temp->next;
            delete temp;
            temp = temp2;
        }
        head = tail = NULL;
    }
private:
	node *head, *tail;
};